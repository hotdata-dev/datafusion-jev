# datafusion-jev

Ask questions about your text from SQL. `prompt_jev` sends each row to the Jev
inference service and returns a typed answer: a probability, a choice from a list
you define, or a score on a scale you define.

Works with DataFusion 55. You bring the HTTP client and API key; the crate handles
the SQL syntax, batching, retries, and result types.

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

## Setup

Add the crate to your project, then follow three steps.

**1. Provide the HTTP client.** Implement `JevClient` with whatever HTTP library
you already use. POST the request body as JSON to
`https://api.typesafe.ai/v1/systemone` with your API key in a header, and return
the parsed JSON response. Keep the key on the server; it never appears in SQL.

```rust
use datafusion_jev::{JevClient, RequestError};

#[derive(Debug)]              // must not print the API key
struct MyClient { /* http client, api key */ }

#[async_trait::async_trait]
impl JevClient for MyClient {
    async fn request(&self, body: serde_json::Value) -> Result<serde_json::Value, RequestError> {
        // Send `body`. Map the outcome:
        //   2xx           -> Ok(parsed JSON)
        //   5xx / network -> Err(RequestError::Unavailable)   (retried)
        //   4xx / other   -> Err(RequestError::Fatal(reason))  (fails the query)
        todo!()
    }
}
```

**2. Register it once per session.**

```rust
let ctx = SessionContext::new();
datafusion_jev::register(&ctx, Arc::new(MyClient::new()));
```

**3. Run queries through `datafusion_jev::sql`.**

```rust
let df = datafusion_jev::sql(&ctx, "SELECT prompt_jev(body, 'Is this urgent?') FROM tickets").await?;
```

Use this in place of `ctx.sql`. It accepts the same SQL and returns the same
`DataFrame`, and `sql_with_options` takes DataFusion's `SQLOptions` just like
`ctx.sql_with_options`. Plain `ctx.sql` does not understand the `prompt_jev`
options syntax and will fail on it.

If your application already parses SQL itself, call `datafusion_jev::rewrite` on
the parsed statement before planning it instead.

## Writing queries

`prompt_jev(text, 'question', options...)`. The text can be any column or
expression. The question and all options must be literals.

### Yes/no probability (default)

```sql
SELECT prompt_jev(body, 'Is the customer asking for a refund?') AS p FROM tickets
```

Returns a `DOUBLE` from 0 to 1. Use it directly in filters:

```sql
WHERE prompt_jev(body, 'Is this urgent?') > 0.8
```

Optionally describe what `true` and `false` mean:

```sql
noul := [{label: 'true', description: 'Explicitly requests money back'},
         {label: 'false', description: 'Anything else'}]
```

### Pick one option

```sql
SELECT prompt_jev(body, 'What is this about?', choice := ['billing', 'technical', 'sales']) AS c
FROM tickets
```

Returns a struct. Read its fields with dot syntax:

| Field | Type | Meaning |
|---|---|---|
| `c.choice` | `VARCHAR` | The selected label |
| `c.confidence` | `DOUBLE` | How sure the model is, 0 to 1 |
| `c.probabilities` | list of `{value, probability}` | One entry per option, in the order you listed them |

Two to 255 options. Labels must be unique. Each option can be a plain string or
`{label: '...', description: '...'}`.

### Score on a scale

```sql
SELECT prompt_jev(body, 'How severe is the problem?', score := ['low', 'medium', 'high']) AS s
FROM tickets
```

Returns a struct like `choice`, except the answer is a number:

| Field | Type | Meaning |
|---|---|---|
| `s.score` | `DOUBLE` | Expected position on the scale. With three levels, 0 = low, 2 = high, 1.5 = between medium and high |
| `s.confidence` | `DOUBLE` | How sure the model is, 0 to 1 |
| `s.probabilities` | list of `{index, value, probability}` | One entry per level, in order |

Two to 10 levels, listed from lowest to highest. `choice` and `score` cannot be
combined in one call.

### Batching

```sql
prompt_jev(body, 'question', batch_size := 8)
```

Up to 64 rows are sent per request, default 32. Rows in the same request share
context, which can occasionally influence answers. Use `batch_size := 1` when
each row must be judged in complete isolation.

## What to expect

- **NULL text** returns NULL without calling the service.
- **Repeated text** is asked once and the answer is copied to every matching row.
  Dedup works within each batch of up to 256 rows, so a constant like
  `prompt_jev('hello', ...)` costs about one request per 256 rows scanned.
- **Repeated calls** with identical arguments in one query are evaluated once.
  Reading several fields from one result does not repeat the request.
- **Anywhere in a query.** `prompt_jev` works in `SELECT`, `WHERE`, `ORDER BY`,
  `GROUP BY`, `HAVING`, window `OVER (...)` clauses, and join conditions. A call
  in a join condition must use columns from one side of the join only.
- **Filters run first.** Rows removed by `WHERE` are never sent to the service,
  including when the call sits inside a CTE or subquery.
- **Limits.** Each row's text may be up to 64 KiB after JSON encoding. Longer text
  fails the query. Requests are capped at 256 KiB.
- **Outages.** Each request is tried three times with a 30-second timeout. If all
  fail, the affected rows return NULL and the query completes. Bad credentials,
  invalid options, or a malformed reply fail the query instead.
- **Concurrency.** At most eight requests are in flight at a time per registered
  client, across all queries.
- **Cancelling** a query cancels its in-flight requests.
- **No caching.** Every query calls the service again. Store results in a table
  when you want to reuse them.
- **Dialects.** The `:=` and `{label: ...}` syntax works with DataFusion's default
  dialect and with DuckDB. If your session is set to PostgreSQL or MySQL,
  `prompt_jev` queries fail with an error that says so.

Text columns may be `Utf8`, `LargeUtf8`, `Utf8View`, or dictionary-encoded.

## Compatibility

The SQL interface follows MotherDuck's public `prompt_jev` for the supported
subset. This is not MotherDuck code. Multi-question calls (`questions := ...`)
and the JSON configuration form are not implemented yet.

## How it works

`prompt_jev` is rewritten at parse time into a private async scalar function
whose options are a validated constant. Inference runs through DataFusion's
`AsyncScalarUDF`, so no blocking threads are involved. A physical optimizer rule
collapses repeated identical calls in one projection into a single evaluation.

Your application stays in charge of credentials, who may run paid inference,
endpoint configuration, and usage accounting.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Tests run real DataFusion plans against a mock service. No network access or
credentials are needed.
