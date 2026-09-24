//! The internal function name and the predicates that recognise a call to it,
//! shared by the SQL rewrite, the UDF, and the plan rules.
use datafusion::{
    logical_expr::Expr,
    physical_expr::{PhysicalExpr, ScalarFunctionExpr, async_scalar_function::AsyncFuncExpr},
    physical_plan::async_func::AsyncFuncExec,
};
use std::sync::Arc;

/// The name a `prompt_jev` call is rewritten to before planning. Private to the
/// crate: callers never write it, and it must match everywhere it appears.
pub(crate) const INTERNAL_FUNCTION: &str = "__datafusion_jev";

/// A logical call to the internal function.
pub(crate) fn is_jev(expr: &Expr) -> bool {
    matches!(expr, Expr::ScalarFunction(f) if f.func.name() == INTERNAL_FUNCTION)
}

/// A physical call to the internal function.
pub(crate) fn is_jev_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<ScalarFunctionExpr>()
        .is_some_and(|f| f.fun().name() == INTERNAL_FUNCTION)
}

/// The async expressions of a node. DataFusion deprecated this accessor without
/// a replacement and both physical rules need it, so the exemption lives here
/// rather than at each call site.
pub(crate) fn async_exprs(node: &AsyncFuncExec) -> &[Arc<AsyncFuncExpr>] {
    #[allow(deprecated)]
    node.async_exprs()
}
