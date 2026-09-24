//! Plan-level behaviour: hoisting calls out of nodes DataFusion cannot plan
//! them in, keeping inference below cheap filters, and evaluating a repeated
//! call once.
use crate::common::{Mock, display, run};
use datafusion::prelude::*;
use std::sync::Arc;

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
#[tokio::test]
async fn order_by_prompt_jev_is_hoisted_and_planned() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT id FROM (VALUES (1,'a'),(2,'b'),(3,'c')) t(id, body) \
         ORDER BY prompt_jev(body, 'Urgent?') DESC, id LIMIT 2",
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    assert_eq!(mock.calls.lock().unwrap().len(), 1, "one batched request");
}
#[tokio::test]
async fn window_over_prompt_jev_is_hoisted_and_planned() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT id, rank() OVER (ORDER BY prompt_jev(body, 'Urgent?') DESC) AS rk \
         FROM (VALUES (1,'a'),(2,'b'),(3,'c')) t(id, body) ORDER BY rk, id",
    )
    .await
    .unwrap();
    let text = display(&result);
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    assert!(text.contains("rk"), "{text}");
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn filter_runs_before_inference_when_a_field_is_read_through_a_cte() {
    // DataFusion's leaf pushdown would otherwise move get_field(k, 'choice') and
    // the call it wraps beneath the filter, sending every row to the provider.
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "WITH c AS (SELECT id, prompt_jev(body, 'Topic?', choice := ['a','b']) AS k \
                    FROM (VALUES (1,'keep'),(2,'drop'),(3,'drop')) t(id, body) WHERE id = 1) \
         SELECT id, k.choice FROM c",
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]["questions"].as_object().unwrap().len(),
        1,
        "only the row that passed the filter is asked about: {}",
        calls[0]
    );
}
#[tokio::test]
async fn leaf_pushdown_still_runs_for_queries_without_prompt_jev() {
    // The guard must not disable DataFusion's optimization for ordinary queries.
    let ctx = SessionContext::new();
    datafusion_jev::register(&ctx, Arc::new(Mock::default()));
    let df = datafusion_jev::sql(
        &ctx,
        "EXPLAIN VERBOSE SELECT s.x FROM (SELECT struct(id AS x) AS s FROM (VALUES (1),(2)) t(id) WHERE id = 1) q",
    )
    .await
    .unwrap();
    let text = display(&df.collect().await.unwrap());
    assert!(text.contains("__datafusion_extracted"), "{text}");
}
#[tokio::test]
async fn group_by_prompt_jev_is_planned() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT round(prompt_jev(body, 'Urgent?'), 1) AS p, count(*) AS n \
         FROM (VALUES (1,'a'),(2,'b'),(3,'c')) t(id, body) GROUP BY 1 ORDER BY 1",
    )
    .await
    .unwrap();
    assert!(result.iter().map(|b| b.num_rows()).sum::<usize>() >= 1);
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn one_sided_join_condition_with_prompt_jev_is_hoisted() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT a.id FROM (VALUES (1,'x'),(2,'y')) a(id, body) JOIN (VALUES (1),(2)) b(id) \
         ON a.id = b.id AND prompt_jev(a.body, 'Urgent?') > 0.5 ORDER BY a.id",
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    assert_eq!(mock.calls.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn two_sided_join_condition_is_a_clear_planning_error() {
    let mock = Arc::new(Mock::default());
    let err = run(
        mock.clone(),
        "SELECT a.id FROM (VALUES (1,'x')) a(id, body) JOIN (VALUES (1,'y')) b(id, body) \
         ON a.id = b.id AND prompt_jev(a.body || b.body, 'Same?') > 0.5",
    )
    .await
    .err()
    .unwrap()
    .to_string();
    assert!(err.contains("one side only"), "{err}");
    assert!(!err.contains("called directly"), "{err}");
    assert!(mock.calls.lock().unwrap().is_empty());
}
#[tokio::test]
async fn cheap_conjuncts_filter_rows_before_inference() {
    // `WHERE id = 1 AND prompt_jev(...) > 0.5` must not send rows 2 and 3.
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT id FROM (VALUES (1,'keep'),(2,'drop'),(3,'drop')) t(id, body) \
         WHERE id = 1 AND prompt_jev(body, 'Urgent?') > 0.5",
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let calls = mock.calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]["questions"].as_object().unwrap().len(),
        1,
        "only the row passing the cheap predicate is asked about: {}",
        calls[0]
    );
}
#[tokio::test]
async fn cheap_conjunct_in_a_count_filter_runs_first() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT count(*) AS n FROM (VALUES ('a','x'),('b','y'),('c','z')) t(name, body) \
         WHERE name <> 'c' AND prompt_jev(body, 'Urgent?') > 0.5",
    )
    .await
    .unwrap();
    assert!(display(&result).contains('2'), "{}", display(&result));
    let calls = mock.calls.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .map(|c| c["questions"].as_object().unwrap().len())
            .sum::<usize>(),
        2
    );
}
#[tokio::test]
async fn a_folded_limit_survives_the_filter_split() {
    // With one partition, LimitPushdown folds LIMIT into the FilterExec's fetch.
    // The split must keep it, or the query returns every matching row.
    let mock = Arc::new(Mock::default());
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    datafusion_jev::register(&ctx, mock.clone());
    let df = datafusion_jev::sql(
        &ctx,
        "SELECT id FROM (VALUES (1,'a'),(2,'b'),(3,'c'),(4,'d')) t(id, body) \
         WHERE id > 1 AND prompt_jev(body, 'Urgent?') > 0.5 LIMIT 1",
    )
    .await
    .unwrap();
    let result = df.collect().await.unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    let asked: usize = mock
        .calls
        .lock()
        .unwrap()
        .iter()
        .map(|c| c["questions"].as_object().unwrap().len())
        .sum();
    assert!(
        asked <= 3,
        "rows failing `id > 1` must not be asked about: {asked}"
    );
}
