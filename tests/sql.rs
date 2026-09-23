use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, DictionaryArray, Float64Array, ListArray, RecordBatch, StringArray,
            StructArray,
        },
        datatypes::{Field, Int32Type, Schema, UInt32Type},
    },
    common::Result,
    datasource::MemTable,
    prelude::*,
};
use datafusion_jev::{JevClient, RequestError};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[derive(Debug, Default)]
struct Mock {
    calls: Mutex<Vec<Value>>,
    unavailable: bool,
    fatal: bool,
    malformed: bool,
    /// When set, every choice probability takes this value instead of a
    /// one-hot distribution, so a test can control the probability sum.
    probability: Option<f64>,
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
async fn run(client: Arc<dyn JevClient>, sql: &str) -> Result<Vec<RecordBatch>> {
    plan_and_run(SessionContext::new(), client, sql).await
}
/// Run `sql` against a single-column table `t(body)` built from `body`.
async fn run_table(
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
async fn plan_and_run(
    ctx: SessionContext,
    client: Arc<dyn JevClient>,
    sql: &str,
) -> Result<Vec<RecordBatch>> {
    datafusion_jev::register(&ctx, client);
    datafusion_jev::sql(&ctx, sql).await?.collect().await
}
fn display(batches: &[RecordBatch]) -> String {
    datafusion::arrow::util::pretty::pretty_format_batches(batches)
        .unwrap()
        .to_string()
}
#[tokio::test]
async fn exact_user_syntax_and_typed_fields() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        r#"
      WITH classified AS (
        SELECT conversation_id, prompt_jev(transcript, 'Identify the customer''s main complaint', choice := [
          {label: 'billing', description: 'Payments, invoices, and refunds'},
          {label: 'technical', description: 'Errors, outages, and integrations'},
          {label: 'sales', description: 'Pricing and upgrades'},
          {label: 'account', description: 'Cancellations and account administration'}
        ]) AS classification
        FROM (VALUES (1, 'Please refund this payment')) t(conversation_id, transcript)
      ) SELECT conversation_id, classification.choice, classification.confidence, classification.probabilities FROM classified
    "#,
    )
    .await
    .unwrap();
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
    let result = run(
        mock.clone(),
        "SELECT id, prompt_jev(body, 'Urgent?', batch_size := 2) AS p \
         FROM (VALUES (1,'a'), (2,NULL), (3,'b'), (4,'c')) t(id,body)",
    )
    .await
    .unwrap();
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
    let result = run(
        Arc::new(Mock::default()),
        "SELECT prompt_jev('bad', 'Severity?', score := ['low','medium','high']) AS s, \
         prompt_jev('bad','Urgent?') AS p",
    )
    .await
    .unwrap();
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
    let result = run(
        mock,
        "SELECT count(*) AS n FROM (VALUES ('a'),('b')) t(body) \
         WHERE prompt_jev(body,'Urgent?') > 0.8",
    )
    .await
    .unwrap();
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

#[tokio::test]
async fn identical_inputs_are_requested_once() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT prompt_jev(body,'q') AS p FROM (VALUES ('a'),('a'),('b'),('a')) t(body)",
    )
    .await
    .unwrap();
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 1, "repeated text must be asked about once");
    assert_eq!(calls[0]["questions"].as_object().unwrap().len(), 2);
    let column = result[0].column_by_name("p").unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 4);
    assert_eq!(column.null_count(), 0, "every row keeps an answer");
}
#[tokio::test]
async fn constant_input_is_requested_once() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT prompt_jev('x','q') AS p FROM (VALUES (1),(2),(3)) t(id)",
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["questions"].as_object().unwrap().len(), 1);
}
fn many_labels(n: usize) -> String {
    (0..n)
        .map(|i| format!("'c{i}'"))
        .collect::<Vec<_>>()
        .join(",")
}
#[tokio::test]
async fn probability_sum_tolerance_scales_with_criteria_count() {
    // 200 criteria rounded to three decimals can drift far from one.
    let mock = Arc::new(Mock {
        probability: Some(0.004),
        ..Default::default()
    });
    let sql = format!(
        "SELECT prompt_jev('a','q',choice := [{}]) AS p",
        many_labels(200)
    );
    run(mock, &sql).await.expect("rounding drift is tolerated");
    // Garbage is still rejected: 200 * 0.0075 = 1.5.
    let mock = Arc::new(Mock {
        probability: Some(0.0075),
        ..Default::default()
    });
    let sql = format!(
        "SELECT prompt_jev('a','q',choice := [{}]) AS p",
        many_labels(200)
    );
    assert!(run(mock, &sql).await.is_err());
    // A small question keeps the tight tolerance.
    let mock = Arc::new(Mock {
        probability: Some(0.4),
        ..Default::default()
    });
    assert!(
        run(mock, "SELECT prompt_jev('a','q',choice := ['x','y']) AS p")
            .await
            .is_err()
    );
}
#[tokio::test]
async fn oversized_encoded_input_is_rejected_without_batch_size_advice() {
    let mock = Arc::new(Mock::default());
    // ~60k control characters: six bytes each once JSON-encoded.
    let body = Arc::new(StringArray::from(vec!["\u{1}".repeat(60_000)])) as ArrayRef;
    let error = run_table(
        mock.clone(),
        "SELECT prompt_jev(body,'q',batch_size := 1) AS p FROM t",
        body,
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(error.contains("input exceeds"), "{error}");
    assert!(!error.contains("batch_size"), "{error}");
    assert!(mock.calls.lock().unwrap().is_empty());
}
#[tokio::test]
async fn dictionary_encoded_input_is_accepted() {
    let mock = Arc::new(Mock::default());
    let body = Arc::new(
        vec!["alpha", "beta", "alpha"]
            .into_iter()
            .collect::<DictionaryArray<Int32Type>>(),
    ) as ArrayRef;
    let result = run_table(
        mock.clone(),
        "SELECT prompt_jev(body,'q') AS p FROM t",
        body,
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    assert_eq!(result[0].column_by_name("p").unwrap().null_count(), 0);
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["questions"].as_object().unwrap().len(), 2);
}
#[tokio::test]
async fn batches_split_on_encoded_size() {
    let mock = Arc::new(Mock::default());
    // Three distinct ~20 KiB rows: the 32 KiB batch budget allows only one each.
    let body = Arc::new(StringArray::from(
        (0..3)
            .map(|i| format!("{i}{}", "x".repeat(20_000)))
            .collect::<Vec<_>>(),
    )) as ArrayRef;
    run_table(
        mock.clone(),
        "SELECT prompt_jev(body,'q') AS p FROM t",
        body,
    )
    .await
    .unwrap();
    let calls = mock.calls.lock().unwrap();
    assert_eq!(
        calls.len(),
        3,
        "default batch_size must still split by size"
    );
    assert!(calls.iter().all(|c| c["state"].is_string()));
}
#[tokio::test]
async fn repeated_expressions_share_one_request_and_keep_columns() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "WITH c AS (SELECT id, \
           prompt_jev(body,'a',choice := ['alpha','beta']) AS x, \
           prompt_jev(body,'b') AS y, \
           prompt_jev(body,'a',choice := ['alpha','beta']) AS z, \
           body FROM (VALUES (1,'m'),(2,'n')) t(id,body)) \
         SELECT id, x.choice AS xc, y, z.choice AS zc, body FROM c",
    )
    .await
    .unwrap();
    assert_eq!(
        mock.calls.lock().unwrap().len(),
        2,
        "the repeated expression must not be evaluated twice"
    );
    let names: Vec<_> = result[0]
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(names, ["id", "xc", "y", "zc", "body"]);
    let text = display(&result);
    for expected in ["alpha", "0.9", "| m ", "| n "] {
        assert!(text.contains(expected), "{expected} missing from {text}");
    }
    let xc = result[0].column_by_name("xc").unwrap();
    let zc = result[0].column_by_name("zc").unwrap();
    assert_eq!(xc.as_ref(), zc.as_ref(), "both slots hold the same answer");
}
/// The struct field `name` of the `probabilities` list entries, as (value, probability).
fn probabilities(batch: &RecordBatch, name: &str) -> (Vec<String>, Vec<f64>, Option<Vec<u32>>) {
    let column = batch.column_by_name(name).unwrap();
    let outer = column.as_any().downcast_ref::<StructArray>().unwrap();
    let list = outer
        .column_by_name("probabilities")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    let entries = list.value(0);
    let entries = entries.as_any().downcast_ref::<StructArray>().unwrap();
    let values = entries
        .column_by_name("value")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let probs = entries
        .column_by_name("probability")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let index = entries.column_by_name("index").map(|c| {
        c.as_any()
            .downcast_ref::<datafusion::arrow::array::PrimitiveArray<UInt32Type>>()
            .unwrap()
            .values()
            .to_vec()
    });
    (
        values.iter().map(|v| v.unwrap().to_owned()).collect(),
        probs.values().to_vec(),
        index,
    )
}
#[tokio::test]
async fn probabilities_follow_caller_order() {
    let result = run(
        Arc::new(Mock::default()),
        "SELECT prompt_jev('a','q',choice := ['billing','technical','sales']) AS c, \
         prompt_jev('a','q',score := ['low','medium','high']) AS s",
    )
    .await
    .unwrap();
    let (labels, probs, index) = probabilities(&result[0], "c");
    assert_eq!(labels, ["billing", "technical", "sales"]);
    assert_eq!(probs, [1.0, 0.0, 0.0]);
    assert_eq!(index, None, "choice carries no index");
    let (labels, probs, index) = probabilities(&result[0], "s");
    assert_eq!(labels, ["low", "medium", "high"]);
    assert_eq!(probs, [0.25, 0.0, 0.75]);
    assert_eq!(index, Some(vec![0, 1, 2]));
}
#[tokio::test]
async fn score_criteria_carry_their_labels() {
    let mock = Arc::new(Mock::default());
    run(
        mock.clone(),
        "SELECT prompt_jev('a','q',score := [\
           {label: 'low', description: 'minor annoyance'}, \
           {label: 'high', description: 'outage'}]) AS s",
    )
    .await
    .unwrap();
    let calls = mock.calls.lock().unwrap();
    let criteria = calls[0]["questions"]["row_0"]["criteria"]
        .as_array()
        .unwrap();
    assert_eq!(
        criteria,
        &vec![json!("low: minor annoyance"), json!("high: outage")]
    );
}

#[derive(Debug, Default)]
struct HangsOnce {
    mock: Mock,
    calls: std::sync::atomic::AtomicUsize,
}
#[async_trait]
impl JevClient for HangsOnce {
    async fn request(&self, body: Value) -> std::result::Result<Value, RequestError> {
        if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
            std::future::pending::<()>().await;
        }
        self.mock.request(body).await
    }
}
#[tokio::test(start_paused = true)]
async fn timed_out_request_is_retried() {
    let client = Arc::new(HangsOnce::default());
    let result = run(client.clone(), "SELECT prompt_jev('a','q') AS p")
        .await
        .unwrap();
    assert!(display(&result).contains("0.9"));
    assert_eq!(client.calls.load(std::sync::atomic::Ordering::SeqCst), 2);
}

#[tokio::test]
async fn drop_in_sql_handles_ddl_set_and_options() {
    let mock = Arc::new(Mock::default());
    let ctx = SessionContext::new();
    datafusion_jev::register(&ctx, mock.clone());
    datafusion_jev::sql(&ctx, "CREATE TABLE t AS VALUES ('a'), ('b')")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    datafusion_jev::sql(&ctx, "SET datafusion.execution.batch_size = 1024")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let result = datafusion_jev::sql(&ctx, "SELECT prompt_jev(column1, 'q') AS p FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
    let forbidden = datafusion_jev::sql_with_options(
        &ctx,
        "CREATE TABLE u AS VALUES (1)",
        datafusion::execution::context::SQLOptions::new().with_allow_ddl(false),
    )
    .await;
    assert!(forbidden.is_err());
}
#[tokio::test]
async fn strict_dialect_gets_a_clear_error() {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new().set_str("datafusion.sql_parser.dialect", "postgresql"),
    );
    datafusion_jev::register(&ctx, Arc::new(Mock::default()));
    let err = datafusion_jev::sql(&ctx, "SELECT prompt_jev('a','q', choice := ['x','y'])")
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("generic (default) or duckdb"), "{err}");
    assert!(err.contains("postgresql"), "{err}");
    let plain = datafusion_jev::sql(&ctx, "SELECT 1 +")
        .await
        .err()
        .unwrap()
        .to_string();
    assert!(!plain.contains("prompt_jev"), "{plain}");
}
