//! Jev decisions for DataFusion 55. Parse with `rewrite_statement`, then plan
//! using a context configured by `register`. Network IO uses DataFusion's native
//! asynchronous UDF execution; no blocking threads or detached inference tasks.
mod optimizer;
pub mod sql;
mod udf;
use async_trait::async_trait;
use datafusion::prelude::SessionContext;
use serde_json::Value;
pub use sql::rewrite_statement;
use std::sync::Arc;

/// Transport errors never carry input text, credentials, or response bodies.
#[derive(Debug)]
pub enum RequestError {
    /// Retried; exhausted requests produce NULLs for their input rows.
    Unavailable,
    /// Authentication, invalid configuration, or malformed provider responses.
    Fatal(String),
}
/// Supplied by the host so credentials and outbound policy stay host-owned.
#[async_trait]
pub trait JevClient: Send + Sync + std::fmt::Debug {
    async fn request(&self, body: Value) -> Result<Value, RequestError>;
}
/// Register the async implementation. Install once per host/client scope.
/// Public SQL must first pass through `rewrite_statement`.
pub fn register(ctx: &SessionContext, client: Arc<dyn JevClient>) {
    ctx.register_udf(udf::function(client));
    let state_ref = ctx.state_ref();
    let mut state = state_ref.write();
    *state = datafusion::execution::SessionStateBuilder::new_from_existing(state.clone())
        .with_physical_optimizer_rule(Arc::new(optimizer::DeduplicateJev))
        .build();
}

/// Rewrite DataFusion statements, including its EXPLAIN wrapper.
pub fn rewrite(
    statement: &mut datafusion::sql::parser::Statement,
) -> datafusion::common::Result<()> {
    match statement {
        datafusion::sql::parser::Statement::Statement(inner) => rewrite_statement(inner),
        datafusion::sql::parser::Statement::Explain(explain) => rewrite(&mut explain.statement),
        _ => Ok(()),
    }
}
