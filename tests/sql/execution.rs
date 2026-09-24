//! Running queries against the mock provider: result shapes, batching,
//! deduplication, NULL handling, retries, cancellation, and concurrency.
use crate::common::{Mock, display, run, run_table};
use async_trait::async_trait;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, DictionaryArray, Float64Array, ListArray, RecordBatch, StringArray,
        StructArray,
    },
    datatypes::{Int32Type, UInt32Type},
};
use datafusion_jev::{JevClient, RequestError};
use serde_json::{Value, json};
use std::sync::Arc;

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
