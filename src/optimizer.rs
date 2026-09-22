//! DataFusion 55 extracts every occurrence of an async function, including
//! repeated struct field access. Keep one Jev evaluation and project its result
//! into the original slots so all downstream column indices remain valid.
use datafusion::{
    common::{
        Result,
        tree_node::{Transformed, TransformedResult, TreeNode},
    },
    config::ConfigOptions,
    physical_expr::{PhysicalExpr, ScalarFunctionExpr, expressions::Column},
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, async_func::AsyncFuncExec, projection::ProjectionExec},
};
use std::sync::Arc;
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
            let Some(node) = plan.downcast_ref::<AsyncFuncExec>() else {
                return Ok(Transformed::no(plan));
            };
            // DF deprecated this accessor without a replacement. We need it to
            // avoid duplicate paid inference while preserving the output schema.
            #[allow(deprecated)]
            let expressions = node.async_exprs();
            let mut unique: Vec<
                Arc<datafusion::physical_expr::async_scalar_function::AsyncFuncExpr>,
            > = vec![];
            let mut indices = vec![];
            for expr in expressions {
                let is_jev = expr
                    .func
                    .downcast_ref::<ScalarFunctionExpr>()
                    .is_some_and(|f| f.fun().name() == "__datafusion_jev");
                let existing = if is_jev {
                    unique.iter().position(|x| x.func == expr.func.clone())
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
