//! Argument validation, the SQL rewrite itself, dialect handling, name
//! matching, and the plan-time size caps.
use crate::common::{Mock, display, run};
use datafusion::prelude::*;
use std::sync::Arc;

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
async fn explain_does_not_run_inference() {
    let mock = Arc::new(Mock::default());
    run(mock.clone(), "EXPLAIN SELECT prompt_jev('a','q') AS p")
        .await
        .unwrap();
    assert!(mock.calls.lock().unwrap().is_empty());
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
#[tokio::test]
async fn quoted_and_uppercase_names_are_the_same_function() {
    let mock = Arc::new(Mock::default());
    let result = run(
        mock.clone(),
        "SELECT \"prompt_jev\"('a', 'q') AS a, PROMPT_JEV('a', 'q') AS b, \"PROMPT_JEV\"('a', 'q') AS c",
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    assert!(display(&result).contains("0.9"));
}
#[tokio::test]
async fn oversized_instructions_and_labels_fail_at_plan_time() {
    let long_q = "q".repeat(5_000);
    let long_label = "L".repeat(300);
    let long_desc = "d".repeat(2_000);
    for (sql, needle) in [
        (
            format!("SELECT prompt_jev('a', '{long_q}')"),
            "4000 characters",
        ),
        (
            format!("SELECT prompt_jev('a', 'q', choice := ['{long_label}', 'b'])"),
            "labels are limited",
        ),
        (
            format!(
                "SELECT prompt_jev('a', 'q', choice := [{{label: 'a', description: '{long_desc}'}}, 'b'])"
            ),
            "descriptions to 1024",
        ),
    ] {
        let mock = Arc::new(Mock::default());
        let err = run(mock.clone(), &sql).await.err().unwrap().to_string();
        assert!(err.contains(needle), "{needle}: {err}");
        assert!(mock.calls.lock().unwrap().is_empty());
    }
}
