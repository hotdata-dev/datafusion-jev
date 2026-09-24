//! Shared harness for the `prompt_jev` integration tests: a mock provider and
//! helpers that register the crate, run a query, and format the result.
use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{ArrayRef, RecordBatch},
        datatypes::{Field, Schema},
    },
    common::Result,
    datasource::MemTable,
    prelude::*,
};
use datafusion_jev::{JevClient, RequestError};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
pub struct Mock {
    pub calls: Mutex<Vec<Value>>,
    pub unavailable: bool,
    pub fatal: bool,
    pub malformed: bool,
    /// When set, every choice probability takes this value instead of a
    /// one-hot distribution, so a test can control the probability sum.
    pub probability: Option<f64>,
}
#[async_trait]
impl JevClient for Mock {
    async fn request(&self, body: Value) -> std::result::Result<Value, RequestError> {
        self.calls.lock().unwrap().push(body.clone());
        if self.unavailable {
            return Err(RequestError::Unavailable);
        }
        if self.fatal {
            return Err(RequestError::Fatal("HTTP 422".into()));
        }
        if self.malformed {
            return Ok(json!({"answers":{}}));
        }
        let mut answers = serde_json::Map::new();
        for (key, q) in body["questions"].as_object().unwrap() {
            let value = match q["type"].as_str().unwrap() {
                "noul" => json!({"type":"noul","noul":0.9}),
                "choice" => {
                    let labels: Vec<_> = q["criteria"].as_object().unwrap().keys().collect();
                    let probabilities: serde_json::Map<String, Value> = labels
                        .iter()
                        .enumerate()
                        .map(|(i, k)| {
                            let p = self.probability.unwrap_or(if i == 0 { 1.0 } else { 0.0 });
                            ((*k).clone(), json!(p))
                        })
                        .collect();
                    json!({"type":"choice","choice":labels[0],"confidence":0.95,"probabilities":probabilities})
                }
                "score" => {
                    let n = q["criteria"].as_array().unwrap().len();
                    let probabilities: serde_json::Map<String, Value> = (0..n)
                        .map(|i| {
                            (
                                i.to_string(),
                                json!(if i == 0 {
                                    0.25
                                } else if i == n - 1 {
                                    0.75
                                } else {
                                    0.0
                                }),
                            )
                        })
                        .collect();
                    json!({"type":"score","score":0.75*(n-1) as f64,"confidence":0.7,"probabilities":probabilities})
                }
                _ => unreachable!(),
            };
            answers.insert(key.clone(), value);
        }
        Ok(json!({"answers":answers}))
    }
}
pub async fn run(client: Arc<dyn JevClient>, sql: &str) -> Result<Vec<RecordBatch>> {
    plan_and_run(SessionContext::new(), client, sql).await
}
/// Run `sql` against a single-column table `t(body)` built from `body`.
pub async fn run_table(
    client: Arc<dyn JevClient>,
    sql: &str,
    body: ArrayRef,
) -> Result<Vec<RecordBatch>> {
    let ctx = SessionContext::new();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "body",
        body.data_type().clone(),
        true,
    )]));
    let batch = RecordBatch::try_new(schema.clone(), vec![body])?;
    ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;
    plan_and_run(ctx, client, sql).await
}
pub async fn plan_and_run(
    ctx: SessionContext,
    client: Arc<dyn JevClient>,
    sql: &str,
) -> Result<Vec<RecordBatch>> {
    datafusion_jev::register(&ctx, client);
    datafusion_jev::sql(&ctx, sql).await?.collect().await
}
pub fn display(batches: &[RecordBatch]) -> String {
    datafusion::arrow::util::pretty::pretty_format_batches(batches)
        .unwrap()
        .to_string()
}
