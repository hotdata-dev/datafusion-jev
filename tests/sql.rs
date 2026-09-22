use async_trait::async_trait;
use datafusion::{common::Result, dataframe::DataFrame, prelude::*};
use datafusion_jev::{JevClient, RequestError};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct Mock {
    calls: Mutex<Vec<Value>>,
    unavailable: bool,
    fatal: bool,
    malformed: bool,
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
                        .map(|(i, k)| ((*k).clone(), json!(if i == 0 { 1.0 } else { 0.0 })))
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
async fn run(
    client: Arc<dyn JevClient>,
    sql: &str,
) -> Result<Vec<datafusion::arrow::array::RecordBatch>> {
    let ctx = SessionContext::new();
    datafusion_jev::register(&ctx, client);
    let state = ctx.state();
    let mut stmt = datafusion::sql::parser::DFParserBuilder::new(sql)
        .with_dialect(&datafusion::sql::sqlparser::dialect::DuckDbDialect {})
        .build()?
        .parse_statements()?
        .pop_front()
        .unwrap();
    datafusion_jev::rewrite(&mut stmt)?;
    let plan = state.statement_to_plan(stmt).await?;
    DataFrame::new(state, plan).collect().await
}
fn display(batches: &[datafusion::arrow::array::RecordBatch]) -> String {
    datafusion::arrow::util::pretty::pretty_format_batches(batches)
        .unwrap()
        .to_string()
}
#[tokio::test]
async fn exact_user_syntax_and_typed_fields() {
    let mock = Arc::new(Mock::default());
    let result=run(mock.clone(),r#"
      WITH classified AS (
        SELECT conversation_id, prompt_jev(transcript, 'Identify the customer''s main complaint', choice := [
          {label: 'billing', description: 'Payments, invoices, and refunds'},
          {label: 'technical', description: 'Errors, outages, and integrations'},
          {label: 'sales', description: 'Pricing and upgrades'},
          {label: 'account', description: 'Cancellations and account administration'}
        ]) AS classification
        FROM (VALUES (1, 'Please refund this payment')) t(conversation_id, transcript)
      ) SELECT conversation_id, classification.choice, classification.confidence, classification.probabilities FROM classified
    "#).await.unwrap();
    let text = display(&result);
    assert!(text.contains("0.95"), "{text}");
    assert!(text.contains("account"), "{text}");
    assert_eq!(
        mock.calls.lock().unwrap().len(),
        1,
        "reading fields must not repeat inference"
    );
}
#[tokio::test]
async fn batching_and_null_input() {
    let mock = Arc::new(Mock::default());
    let result=run(mock.clone(),"SELECT id, prompt_jev(body, 'Urgent?', batch_size := 2) AS p FROM (VALUES (1,'a'), (2,NULL), (3,'b'), (4,'c')) t(id,body)").await.unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(
        calls
            .iter()
            .map(|v| v["questions"].as_object().unwrap().len())
            .sum::<usize>(),
        3
    );
    assert!(display(&result).contains("0.9"));
}
#[tokio::test]
async fn score_and_noul_types() {
    let result=run(Arc::new(Mock::default()),"SELECT prompt_jev('bad', 'Severity?', score := ['low','medium','high']) AS s, prompt_jev('bad','Urgent?') AS p").await.unwrap();
    let text = display(&result);
    assert!(text.contains("1.5"), "{text}");
    assert!(text.contains("0.9"), "{text}");
    assert!(text.contains("medium"), "{text}");
}
#[tokio::test]
async fn invalid_arguments_never_call_provider() {
    for sql in [
        "SELECT prompt_jev('a','q',choice := ['a'])",
        "SELECT prompt_jev('a','q',choice := ['a','a'])",
        "SELECT prompt_jev('a','q',choice := ['a','b'], score := ['x','y'])",
        "SELECT prompt_jev('a','q',batch_size := 0)",
        "SELECT prompt_jev('a','q',batch_size := 65)",
        "SELECT prompt_jev('a',body) FROM (VALUES ('q')) t(body)",
        "SELECT prompt_jev('a','q',choice := body) FROM (VALUES ('q')) t(body)",
        "SELECT prompt_jev('a','q',noul := ['yes','no'])",
        "SELECT prompt_jev('a','q',model := 'foo')",
        "SELECT prompt_jev(42,'q')",
        "SELECT prompt_jev('a','q',noul := [])",
    ] {
        let mock = Arc::new(Mock::default());
        assert!(run(mock.clone(), sql).await.is_err(), "{sql}");
        assert!(mock.calls.lock().unwrap().is_empty());
    }
}
#[tokio::test]
async fn unavailable_is_retried_then_null() {
    let mock = Arc::new(Mock {
        unavailable: true,
        ..Default::default()
    });
    let result = run(mock.clone(), "SELECT prompt_jev('a','q') AS p")
        .await
        .unwrap();
    assert!(result[0].column(0).is_null(0));
    assert_eq!(mock.calls.lock().unwrap().len(), 3);
}
#[tokio::test]
async fn provider_validation_and_malformed_answers_fail() {
    for mock in [
        Mock {
            fatal: true,
            ..Default::default()
        },
        Mock {
            malformed: true,
            ..Default::default()
        },
    ] {
        let mock = Arc::new(mock);
        assert!(
            run(mock.clone(), "SELECT prompt_jev('a','q') AS p")
                .await
                .is_err()
        );
        assert_eq!(mock.calls.lock().unwrap().len(), 1);
    }
}
#[tokio::test]
async fn nulls_and_empty_results_make_no_requests() {
    let mock = Arc::new(Mock::default());
    run(mock.clone(), "SELECT prompt_jev(NULL,'q') AS p")
        .await
        .unwrap();
    run(
        mock.clone(),
        "SELECT prompt_jev(body,'q') AS p FROM (VALUES ('a')) t(body) WHERE false",
    )
    .await
    .unwrap();
    assert!(mock.calls.lock().unwrap().is_empty());
}
#[tokio::test]
async fn semantic_filter_and_aggregate() {
    let mock = Arc::new(Mock::default());
    let result=run(mock,"SELECT count(*) AS n FROM (VALUES ('a'),('b')) t(body) WHERE prompt_jev(body,'Urgent?') > 0.8").await.unwrap();
    assert!(display(&result).contains("2"));
}

#[derive(Debug)]
struct Hanging {
    started: Arc<tokio::sync::Notify>,
    dropped: Arc<tokio::sync::Notify>,
}
struct DropSignal(Arc<tokio::sync::Notify>);
impl Drop for DropSignal {
    fn drop(&mut self) {
        self.0.notify_one();
    }
}
#[async_trait]
impl JevClient for Hanging {
    async fn request(&self, _: Value) -> std::result::Result<Value, RequestError> {
        let _signal = DropSignal(self.dropped.clone());
        self.started.notify_one();
        std::future::pending().await
    }
}
#[tokio::test]
async fn cancelling_query_drops_inflight_request() {
    let started = Arc::new(tokio::sync::Notify::new());
    let dropped = Arc::new(tokio::sync::Notify::new());
    let client = Arc::new(Hanging {
        started: started.clone(),
        dropped: dropped.clone(),
    });
    let task = tokio::spawn(async move { run(client, "SELECT prompt_jev('a','q') AS p").await });
    tokio::time::timeout(std::time::Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    task.abort();
    let _ = task.await;
    tokio::time::timeout(std::time::Duration::from_secs(5), dropped.notified())
        .await
        .unwrap();
}
#[tokio::test]
async fn isolated_batches_keep_state_as_single_text() {
    let mock = Arc::new(Mock::default());
    run(
        mock.clone(),
        "SELECT prompt_jev(body,'q',batch_size := 1) FROM (VALUES ('alpha'),('beta')) t(body)",
    )
    .await
    .unwrap();
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|c| c["state"].is_string()));
}
#[tokio::test]
async fn explain_does_not_run_inference() {
    let mock = Arc::new(Mock::default());
    run(mock.clone(), "EXPLAIN SELECT prompt_jev('a','q') AS p")
        .await
        .unwrap();
    assert!(mock.calls.lock().unwrap().is_empty());
}

#[derive(Debug, Default)]
struct Concurrent {
    mock: Mock,
    active: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl JevClient for Concurrent {
    async fn request(&self, body: Value) -> std::result::Result<Value, RequestError> {
        use std::sync::atomic::Ordering::SeqCst;
        let active = self.active.fetch_add(1, SeqCst) + 1;
        self.peak.fetch_max(active, SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let result = self.mock.request(body).await;
        self.active.fetch_sub(1, SeqCst);
        result
    }
}
#[tokio::test]
async fn inference_concurrency_is_bounded() {
    let mock = Arc::new(Concurrent::default());
    let values = (0..64)
        .map(|i| format!("('row {i}')"))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("SELECT prompt_jev(body,'q',batch_size := 1) FROM (VALUES {values}) t(body)");
    run(mock.clone(), &sql).await.unwrap();
    let peak = mock.peak.load(std::sync::atomic::Ordering::SeqCst);
    assert!(peak > 1 && peak <= 8, "peak requests: {peak}");
    assert_eq!(mock.mock.calls.lock().unwrap().len(), 64);
}
