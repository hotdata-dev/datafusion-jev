//! Logical-plan adjustments so `prompt_jev` works wherever DataFusion 55 does
//! not natively plan async functions, and is never evaluated on rows a filter
//! discards.
//!
//! Two problems, two rules:
//!
//! * DataFusion only lifts async calls out of `Projection` and `Filter`. A call
//!   in an `ORDER BY`, a window `OVER (...)`, a `GROUP BY`, or a join condition
//!   reaches the synchronous invoke path and fails with "async functions should
//!   not be called directly". [`HoistJev`] moves such calls into a projection
//!   beneath the node and refers to them by column.
//! * DataFusion's leaf-expression pushdown moves `get_field(k, 'x')` through a
//!   filter and substitutes `k`'s definition without rechecking placement, so a
//!   `prompt_jev` call ends up below the filter and runs on every row.
//!   [`LeafPushdownGuard`] skips those two rules for any plan that contains the
//!   call, and leaves every other plan alone.
use datafusion::{
    common::{
        DFSchema, Result, plan_datafusion_err,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    config::ConfigOptions,
    logical_expr::{Aggregate, Expr, Join, LogicalPlan, LogicalPlanBuilder, Sort, Window, col},
    optimizer::{AnalyzerRule, ApplyOrder, OptimizerConfig, OptimizerRule},
};
use std::sync::Arc;

pub const FUNCTION_NAME: &str = "__datafusion_jev";
const HOIST_PREFIX: &str = "__jev_hoist_";

fn is_jev(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == FUNCTION_NAME)
}

fn contains_jev(expr: &Expr) -> bool {
    expr.exists(|e| Ok(is_jev(e))).unwrap_or(false)
}

fn plan_contains_jev(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = plan.apply_with_subqueries(|p| {
        if p.expressions().iter().any(contains_jev) {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// Runs a wrapped optimizer rule only for plans that do not call `prompt_jev`.
#[derive(Debug)]
pub struct LeafPushdownGuard {
    inner: Arc<dyn OptimizerRule + Send + Sync>,
}

impl LeafPushdownGuard {
    /// Rules that move expressions toward leaves and, in DataFusion 55, carry
    /// an async call along with them.
    pub const GUARDED: &[&str] = &["extract_leaf_expressions", "push_down_leaf_projections"];

    pub fn wrap(
        inner: Arc<dyn OptimizerRule + Send + Sync>,
    ) -> Arc<dyn OptimizerRule + Send + Sync> {
        Arc::new(Self { inner })
    }
}

impl OptimizerRule for LeafPushdownGuard {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn apply_order(&self) -> Option<ApplyOrder> {
        // The guard needs the whole plan to decide, so it drives the traversal
        // itself and replays the inner rule's own order below.
        None
    }
    fn supports_rewrite(&self) -> bool {
        true
    }
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if plan_contains_jev(&plan) {
            return Ok(Transformed::no(plan));
        }
        match self.inner.apply_order() {
            Some(ApplyOrder::TopDown) => {
                plan.transform_down_with_subqueries(|p| self.inner.rewrite(p, config))
            }
            Some(ApplyOrder::BottomUp) => {
                plan.transform_up_with_subqueries(|p| self.inner.rewrite(p, config))
            }
            None => self.inner.rewrite(plan, config),
        }
    }
}

/// Moves `prompt_jev` calls out of sort keys and window expressions into a
/// projection beneath, so DataFusion plans them as ordinary async projections.
#[derive(Debug)]
pub struct HoistJev;

impl AnalyzerRule for HoistJev {
    fn name(&self) -> &str {
        "datafusion_jev_hoist"
    }
    fn analyze(&self, plan: LogicalPlan, _: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up_with_subqueries(|p| match p {
            LogicalPlan::Sort(sort) if sort.expr.iter().any(|s| contains_jev(&s.expr)) => {
                hoist_sort(sort).map(Transformed::yes)
            }
            LogicalPlan::Window(window) if window.window_expr.iter().any(contains_jev) => {
                hoist_window(window).map(Transformed::yes)
            }
            LogicalPlan::Aggregate(agg)
                if agg
                    .group_expr
                    .iter()
                    .chain(&agg.aggr_expr)
                    .any(contains_jev) =>
            {
                hoist_aggregate(agg).map(Transformed::yes)
            }
            LogicalPlan::Join(join)
                if join
                    .on
                    .iter()
                    .flat_map(|(l, r)| [l, r])
                    .chain(&join.filter)
                    .any(contains_jev) =>
            {
                hoist_join(join).map(Transformed::yes)
            }
            other => Ok(Transformed::no(other)),
        })
        .map(|t| t.data)
    }
}

/// Collect each distinct call, project it beneath `input` under a generated
/// name, and return the new input with the call-to-column substitutions.
fn hoist_calls(
    input: Arc<LogicalPlan>,
    exprs: &[&Expr],
) -> Result<(LogicalPlan, Vec<(Expr, Expr)>)> {
    hoist_calls_named(input, exprs, HOIST_PREFIX)
}

fn collect_calls(exprs: &[&Expr]) -> Result<Vec<Expr>> {
    let mut calls: Vec<Expr> = vec![];
    for expr in exprs {
        expr.apply(|e| {
            if is_jev(e) && !calls.contains(e) {
                calls.push(e.clone());
                return Ok(TreeNodeRecursion::Jump);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
    }
    Ok(calls)
}

fn hoist_calls_named(
    input: Arc<LogicalPlan>,
    exprs: &[&Expr],
    prefix: &str,
) -> Result<(LogicalPlan, Vec<(Expr, Expr)>)> {
    let calls = collect_calls(exprs)?;
    if calls.is_empty() {
        return Ok((Arc::unwrap_or_clone(input), vec![]));
    }
    let mut projection: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    let mut substitutions = vec![];
    for (i, call) in calls.into_iter().enumerate() {
        let name = format!("{prefix}{i}");
        projection.push(call.clone().alias(&name));
        substitutions.push((call, col(&name)));
    }
    let projected = LogicalPlanBuilder::from(Arc::unwrap_or_clone(input))
        .project(projection)?
        .build()?;
    Ok((projected, substitutions))
}

fn substitute(expr: Expr, substitutions: &[(Expr, Expr)]) -> Result<Expr> {
    expr.transform_down(|e| {
        if let Some((_, replacement)) = substitutions.iter().find(|(call, _)| *call == e) {
            return Ok(Transformed::new(
                replacement.clone(),
                true,
                TreeNodeRecursion::Jump,
            ));
        }
        Ok(Transformed::no(e))
    })
    .map(|t| t.data)
}

fn hoist_sort(sort: Sort) -> Result<LogicalPlan> {
    let Sort { expr, input, fetch } = sort;
    let original: Vec<Expr> = input
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    let keys: Vec<&Expr> = expr.iter().map(|s| &s.expr).collect();
    let (projected, substitutions) = hoist_calls(input, &keys)?;
    let expr = expr
        .into_iter()
        .map(|s| Ok(s.with_expr(substitute(s.expr.clone(), &substitutions)?)))
        .collect::<Result<Vec<_>>>()?;
    let builder = LogicalPlanBuilder::from(projected);
    let builder = match fetch {
        Some(n) => builder.sort_with_limit(expr, Some(n))?,
        None => builder.sort(expr)?,
    };
    // Drop the hoisted columns again so the plan above sees the schema it planned for.
    builder.project(original)?.build()
}

fn hoist_window(window: Window) -> Result<LogicalPlan> {
    let Window {
        input,
        window_expr,
        schema,
    } = window;
    let original: Vec<Expr> = schema.columns().into_iter().map(Expr::Column).collect();
    let refs: Vec<&Expr> = window_expr.iter().collect();
    let (projected, substitutions) = hoist_calls(input, &refs)?;
    let window_expr = window_expr
        .into_iter()
        .map(|e| {
            // Rewriting the expression changes its generated name; keep the
            // original so the projection above still finds its column.
            let name = e.schema_name().to_string();
            Ok(substitute(e, &substitutions)?.alias(name))
        })
        .collect::<Result<Vec<_>>>()?;
    LogicalPlanBuilder::from(projected)
        .window(window_expr)?
        .project(original)?
        .build()
}

/// Substitute and, when that changed the expression, keep its original output
/// name so the plan above still finds the column.
fn substitute_keeping_name(expr: Expr, substitutions: &[(Expr, Expr)]) -> Result<Expr> {
    let name = expr.schema_name().to_string();
    let rewritten = substitute(expr, substitutions)?;
    Ok(if rewritten.schema_name().to_string() == name {
        rewritten
    } else {
        rewritten.alias(name)
    })
}

fn hoist_aggregate(agg: Aggregate) -> Result<LogicalPlan> {
    let Aggregate {
        input,
        group_expr,
        aggr_expr,
        ..
    } = agg;
    let refs: Vec<&Expr> = group_expr.iter().chain(&aggr_expr).collect();
    let (projected, substitutions) = hoist_calls(input, &refs)?;
    let group_expr = group_expr
        .into_iter()
        .map(|e| substitute_keeping_name(e, &substitutions))
        .collect::<Result<Vec<_>>>()?;
    let aggr_expr = aggr_expr
        .into_iter()
        .map(|e| substitute_keeping_name(e, &substitutions))
        .collect::<Result<Vec<_>>>()?;
    LogicalPlanBuilder::from(projected)
        .aggregate(group_expr, aggr_expr)?
        .build()
}

fn belongs_to(expr: &Expr, schema: &DFSchema) -> bool {
    expr.column_refs()
        .into_iter()
        .all(|c| schema.index_of_column(c).is_ok())
}

fn hoist_join(join: Join) -> Result<LogicalPlan> {
    let Join {
        left,
        right,
        on,
        filter,
        join_type,
        join_constraint,
        schema,
        null_equality,
        null_aware,
    } = join;
    let original: Vec<Expr> = schema.columns().into_iter().map(Expr::Column).collect();
    // Every call must be computable on one side. A call over both sides has
    // no input to be projected from; the caller can compute it in a CTE.
    let mut all: Vec<&Expr> = on.iter().flat_map(|(l, r)| [l, r]).collect();
    all.extend(&filter);
    let (mut left_calls, mut right_calls) = (vec![], vec![]);
    for call in collect_calls(&all)? {
        if belongs_to(&call, left.schema()) {
            left_calls.push(call);
        } else if belongs_to(&call, right.schema()) {
            right_calls.push(call);
        } else {
            return Err(plan_datafusion_err!(
                "prompt_jev in a join condition must reference columns from one side only; \
                 compute it in a CTE or subquery first"
            ));
        }
    }
    let left_refs: Vec<&Expr> = left_calls.iter().collect();
    let right_refs: Vec<&Expr> = right_calls.iter().collect();
    let (left, mut substitutions) = hoist_calls_named(left, &left_refs, "__jev_hoist_l")?;
    let (right, right_subs) = hoist_calls_named(right, &right_refs, "__jev_hoist_r")?;
    substitutions.extend(right_subs);
    let on = on
        .into_iter()
        .map(|(l, r)| {
            Ok((
                substitute(l, &substitutions)?,
                substitute(r, &substitutions)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let filter = filter.map(|f| substitute(f, &substitutions)).transpose()?;
    let join = Join::try_new(
        Arc::new(left),
        Arc::new(right),
        on,
        filter,
        join_type,
        join_constraint,
        null_equality,
        null_aware,
    )?;
    // The join now carries the hoisted columns; project back to what was planned.
    LogicalPlanBuilder::from(LogicalPlan::Join(join))
        .project(original)?
        .build()
}
