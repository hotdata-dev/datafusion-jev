//! Physical-plan rules around DataFusion 55's `AsyncFuncExec`, both of which
//! remove inference the query does not need.
//!
//! * [`FilterBeforeJev`] splits a filter that sits above the async node into
//!   the conjuncts that need no inference and the rest, and runs the cheap ones
//!   beneath the node. `WHERE cheap AND prompt_jev(...) > x` is otherwise
//!   planned as one filter above the node, so every row is sent for inference
//!   before either conjunct runs.
//! * [`DeduplicateJev`] keeps one evaluation of each distinct call and projects
//!   its result into the original slots, so downstream column indices stay
//!   valid. DataFusion extracts every occurrence of an async function,
//!   including repeated struct field access.
use crate::names::{async_exprs, is_jev_expr};
use datafusion::{
    common::{
        Result,
        tree_node::{Transformed, TransformedResult, TreeNode},
    },
    config::ConfigOptions,
    physical_expr::{
        PhysicalExpr, async_scalar_function::AsyncFuncExpr, conjunction, expressions::Column,
        split_conjunction, utils::collect_columns,
    },
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ChildrenPropertiesMode, ExecutionPlan, ReplaceChildrenOptions,
        async_func::AsyncFuncExec,
        filter::{FilterExec, FilterExecBuilder},
        projection::ProjectionExec,
        repartition::RepartitionExec,
    },
};
use std::sync::Arc;

/// The async node and the expressions it evaluates, or `None` when `plan` is
/// some other node. Both rules start here.
fn async_node(plan: &Arc<dyn ExecutionPlan>) -> Option<(&AsyncFuncExec, &[Arc<AsyncFuncExpr>])> {
    let node = plan.downcast_ref::<AsyncFuncExec>()?;
    Some((node, async_exprs(node)))
}

#[derive(Debug)]
pub struct FilterBeforeJev;
impl PhysicalOptimizerRule for FilterBeforeJev {
    fn name(&self) -> &str {
        "datafusion_jev_filter_first"
    }
    fn schema_check(&self) -> bool {
        true
    }
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            let Some(filter) = plan.downcast_ref::<FilterExec>() else {
                return Ok(Transformed::no(plan));
            };
            // Distribution rules may have placed repartition nodes between the
            // filter and the async node; look through them and put them back
            // in the same order afterwards.
            let mut passthrough: Vec<Arc<dyn ExecutionPlan>> = vec![];
            let mut cursor = Arc::clone(filter.input());
            while cursor.downcast_ref::<RepartitionExec>().is_some() {
                let child = Arc::clone(cursor.children()[0]);
                passthrough.push(cursor);
                cursor = child;
            }
            let Some((node, async_exprs)) = async_node(&cursor) else {
                return Ok(Transformed::no(plan));
            };
            if !async_exprs.iter().any(|e| is_jev_expr(&e.func)) {
                return Ok(Transformed::no(plan));
            }
            // Columns at or past the async node's input width are async results.
            let input_len = node.input().schema().fields().len();
            let (needs_inference, cheap): (Vec<_>, Vec<_>) = split_conjunction(filter.predicate())
                .into_iter()
                .cloned()
                .partition(|c| {
                    collect_columns(c)
                        .iter()
                        .any(|col| col.index() >= input_len)
                });
            if cheap.is_empty() || needs_inference.is_empty() {
                return Ok(Transformed::no(plan));
            }
            let pre =
                FilterExecBuilder::new(conjunction(cheap), Arc::clone(node.input())).build()?;
            let mut rebuilt: Arc<dyn ExecutionPlan> =
                Arc::new(AsyncFuncExec::try_new(async_exprs.to_vec(), Arc::new(pre))?);
            for wrapper in passthrough.into_iter().rev() {
                rebuilt = wrapper.replace_children(
                    vec![rebuilt],
                    ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
                )?;
            }
            // Carry the original filter's fetch (LimitPushdown may have folded a
            // LIMIT into it), batch size, and selectivity onto the rebuilt one.
            let post = FilterExecBuilder::new(conjunction(needs_inference), rebuilt)
                .with_fetch(filter.fetch())
                .with_batch_size(filter.batch_size())
                .with_default_selectivity(filter.default_selectivity())
                .apply_projection(
                    filter
                        .projection()
                        .as_ref()
                        .map(|p| p.iter().copied().collect()),
                )?
                .build()?;
            Ok(Transformed::yes(Arc::new(post) as Arc<dyn ExecutionPlan>))
        })
        .data()
    }
}
#[derive(Debug)]
pub struct DeduplicateJev;
impl PhysicalOptimizerRule for DeduplicateJev {
    fn name(&self) -> &str {
        "datafusion_jev_deduplicate"
    }
    fn schema_check(&self) -> bool {
        true
    }
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            let Some((node, expressions)) = async_node(&plan) else {
                return Ok(Transformed::no(plan));
            };
            let mut unique: Vec<Arc<AsyncFuncExpr>> = vec![];
            let mut indices = vec![];
            for expr in expressions {
                let existing = if is_jev_expr(&expr.func) {
                    unique.iter().position(|x| x.func.eq(&expr.func))
                } else {
                    None
                };
                let index = existing.unwrap_or_else(|| {
                    unique.push(expr.clone());
                    unique.len() - 1
                });
                indices.push(index);
            }
            if unique.len() == expressions.len() {
                return Ok(Transformed::no(plan));
            }
            let input_len = node.input().schema().fields().len();
            let dedup: Arc<dyn ExecutionPlan> =
                Arc::new(AsyncFuncExec::try_new(unique, node.input().clone())?);
            let schema = dedup.schema();
            let projection: Vec<(Arc<dyn PhysicalExpr>, String)> = plan
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let source = if i < input_len {
                        i
                    } else {
                        input_len + indices[i - input_len]
                    };
                    (
                        Arc::new(Column::new(schema.field(source).name(), source))
                            as Arc<dyn PhysicalExpr>,
                        f.name().clone(),
                    )
                })
                .collect();
            Ok(Transformed::yes(
                Arc::new(ProjectionExec::try_new(projection, dedup)?) as Arc<dyn ExecutionPlan>,
            ))
        })
        .data()
    }
}
