//! `prompt_jev` for DataFusion 55: ask questions about text from SQL and get
//! typed answers.
//!
//! 1. Implement [`JevClient`] with your HTTP client and API key.
//! 2. Call [`register`] once on your `SessionContext`.
//! 3. Run queries with [`sql()`] instead of `SessionContext::sql`.
//!
//! See the README for the SQL syntax and result types.
mod optimizer;
pub mod sql;
mod udf;
use async_trait::async_trait;
use datafusion::{
    common::plan_datafusion_err,
    execution::context::SQLOptions,
    prelude::{DataFrame, SessionContext},
};
use serde_json::Value;
pub use sql::rewrite_statement;
use std::sync::Arc;

/// How a request failed. Never include input text, credentials, or response
/// bodies in these errors; they can surface in query error messages.
#[derive(Debug)]
pub enum RequestError {
    /// Network error, timeout, or 5xx. Retried; if retries run out the
    /// affected rows become NULL and the query still completes.
    Unavailable,
    /// Authentication, invalid request, or malformed response. Fails the query.
    Fatal(String),
}
/// Your HTTP client for the Jev service. POST `body` as JSON to the endpoint
/// with your API key and return the parsed JSON response. The `Debug`
/// implementation must not print credentials.
#[async_trait]
pub trait JevClient: Send + Sync + std::fmt::Debug {
    async fn request(&self, body: Value) -> Result<Value, RequestError>;
}
/// Make `prompt_jev` available in this session. Call once per `SessionContext`.
/// Then run queries with [`sql()`], or call [`rewrite`] before planning if you
/// parse SQL yourself.
pub fn register(ctx: &SessionContext, client: Arc<dyn JevClient>) {
    ctx.register_udf(udf::function(client));
    let state_ref = ctx.state_ref();
    let mut state = state_ref.write();
    *state = datafusion::execution::SessionStateBuilder::new_from_existing(state.clone())
        .with_physical_optimizer_rule(Arc::new(optimizer::DeduplicateJev))
        .build();
}

/// For applications that parse SQL themselves: rewrite `prompt_jev` calls in a
/// parsed statement (including inside EXPLAIN) so DataFusion can plan it.
/// Safe to call on statements without `prompt_jev`, and safe to call twice.
pub fn rewrite(
    statement: &mut datafusion::sql::parser::Statement,
) -> datafusion::common::Result<()> {
    match statement {
        datafusion::sql::parser::Statement::Statement(inner) => rewrite_statement(inner),
        datafusion::sql::parser::Statement::Explain(explain) => rewrite(&mut explain.statement),
        _ => Ok(()),
    }
}

/// Run a SQL query that may use `prompt_jev`. Use in place of
/// `SessionContext::sql`; same input, same `DataFrame` result.
pub async fn sql(ctx: &SessionContext, text: &str) -> datafusion::common::Result<DataFrame> {
    sql_with_options(ctx, text, SQLOptions::new()).await
}

/// [`sql()`] with DataFusion's `SQLOptions`, for example to forbid DDL.
/// Use in place of `SessionContext::sql_with_options`.
pub async fn sql_with_options(
    ctx: &SessionContext,
    text: &str,
    options: SQLOptions,
) -> datafusion::common::Result<DataFrame> {
    let state = ctx.state();
    let dialect = state.config().options().sql_parser.dialect;
    // Strict dialects either reject `:=` outright or parse it as an ordinary
    // expression, so the failure can surface from parsing or from the rewrite.
    let hint = |e: datafusion::common::DataFusionError| {
        let lenient = matches!(
            dialect,
            datafusion::config::Dialect::Generic | datafusion::config::Dialect::DuckDB
        );
        if lenient || !text.to_lowercase().contains("prompt_jev") {
            return e;
        }
        plan_datafusion_err!(
            "{e}. prompt_jev syntax (':=' options and '{{label: ...}}' literals) requires \
             the generic (default) or duckdb SQL dialect; the session uses {dialect}"
        )
    };
    let mut statement = state.sql_to_statement(text, &dialect).map_err(hint)?;
    rewrite(&mut statement).map_err(hint)?;
    let plan = state.statement_to_plan(statement).await?;
    options.verify_plan(&plan)?;
    ctx.execute_logical_plan(plan).await
}
