# datafusion-jev

Typed Jev decisions in DataFusion 55 SQL. A standalone extension crate; the host supplies the authenticated HTTP client.

```sql
SELECT conversation_id,
       prompt_jev(transcript, 'Identify the customer''s main complaint',
           choice := [
               {label: 'billing', description: 'Payments, invoices, and refunds'},
               {label: 'technical', description: 'Errors, outages, and integrations'},
               {label: 'sales', description: 'Pricing and upgrades'},
               {label: 'account', description: 'Cancellations and account administration'}
           ]) AS classification
FROM customer_conversations;
```

## SQL interface

- `prompt_jev(text, 'question')`: a nullable `DOUBLE` between 0 and 1 (`noul`).
- `choice := ['a', 'b']`: struct with `choice`, `probabilities`, `confidence`.
- `score := ['low', 'medium', 'high']`: struct with weighted `score` (0–2 here), `probabilities`, `confidence`.
- Criteria also accept `{label: '...', description: '...'}` literals. Descriptions may be NULL.
- `noul := [...]` may attach descriptions to exactly `true` and `false`.
- `batch_size := 1..64`, default 32. Only input varies per row; instructions and options currently require SQL literals, not computed constant expressions.
- Choice requires 2–255 unique labels; score requires 2–10. These modes are mutually exclusive.

Choice probabilities are `STRUCT(value VARCHAR, probability DOUBLE)[]`, in caller order. Score probabilities additionally have `index UINTEGER`. Confidence is the provider's confidence, not a locally inferred value.

This first version supports single-question calls. `questions := ...` and the JSON configuration escape hatch are not implemented. It follows MotherDuck's public interface for the supported subset; it is not MotherDuck code.

## Host integration

1. Implement `JevClient::request(Value)` using your existing HTTP transport. POST the JSON to `https://api.typesafe.ai/v1/systemone`, authenticate with a server-side API key, and return the parsed response. Apply your outbound policy; never put credentials in SQL. Distinguish retryable `Unavailable` from `Fatal` configuration/protocol errors.
2. Call `datafusion_jev::register(&ctx, Arc::new(client))` once per client scope.
3. Parse the user's SQL, call `datafusion_jev::rewrite(&mut datafusion_statement)` (which also handles EXPLAIN), and pass the rewritten AST to `SessionState::statement_to_plan`.

The extension uses DataFusion's `AsyncScalarUDF`; it does not replace the host's query planner. Registration appends a physical optimizer rule that deduplicates repeated Jev expressions within an async execution node, preserving all output slots. SQL lowering preserves the input expression and encodes validated options as a constant. The private `__datafusion_jev` function is an implementation detail.

The host is responsible for credentials, authorization to use paid inference, endpoint configuration, HTTP status classification, usage accounting, and response-size limits. `Debug` implementations of clients must redact credentials. No live requests or credentials are required by this crate's tests.

## Execution and failures

- Up to eight requests execute concurrently per registration, shared across partitions and queries. Futures are awaited inline and are dropped on query cancellation.
- Input is evaluated in bounded batches of at most 256 rows. NULL input skips inference and returns NULL.
- Batches share a state object and use a separately scoped question for each row. This can affect predictions: choose `batch_size := 1` for isolated state per row.
- Input is capped at 64 KiB per row; encoded requests at 256 KiB. Requests use `jev-latest`.
- Transient failures get three attempts with a 30-second timeout per attempt and bounded backoff. Exhaustion returns NULL for affected rows. Invalid configuration, authentication, or malformed answers fail the query.
- No persistent result cache: materialize results using the host's result-to-table facility when you want to reuse them without inference.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Tests execute real DataFusion plans against a mock provider, covering the public syntax, result fields, batching, NULLs, filtering, and error behavior.
