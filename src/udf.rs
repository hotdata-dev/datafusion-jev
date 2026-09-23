use crate::{JevClient, RequestError, sql::Question};
use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, StructArray, new_empty_array},
        datatypes::{DataType, Field, FieldRef, Fields},
    },
    common::{Result, ScalarValue, exec_datafusion_err, plan_datafusion_err},
    logical_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        Volatility,
        async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl},
    },
};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};
use tokio::sync::Semaphore;

#[derive(Debug)]
struct Jev {
    signature: Signature,
    client: Arc<dyn JevClient>,
    permits: Arc<Semaphore>,
}
impl PartialEq for Jev {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.client, &other.client)
    }
}
impl Eq for Jev {}
impl Hash for Jev {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.client) as *const () as usize).hash(state);
    }
}
pub fn function(client: Arc<dyn JevClient>) -> ScalarUDF {
    AsyncScalarUDF::new(Arc::new(Jev {
        signature: Signature::any(2, Volatility::Volatile),
        client,
        permits: Arc::new(Semaphore::new(8)),
    }))
    .into_scalar_udf()
}
fn config(value: &ScalarValue) -> Result<Question> {
    let text = match value {
        ScalarValue::Utf8(Some(s))
        | ScalarValue::Utf8View(Some(s))
        | ScalarValue::LargeUtf8(Some(s)) => s,
        _ => {
            return Err(plan_datafusion_err!(
                "prompt_jev configuration must be constant"
            ));
        }
    };
    let q: Question = serde_json::from_str(text)
        .map_err(|_| plan_datafusion_err!("invalid prompt_jev configuration"))?;
    q.validate()?;
    Ok(q)
}
fn fields(q: &Question) -> Fields {
    let mut p = vec![];
    if q.kind == "score" {
        p.push(Field::new("index", DataType::UInt32, false));
    }
    p.push(Field::new("value", DataType::Utf8, false));
    p.push(Field::new("probability", DataType::Float64, false));
    vec![
        Field::new(
            if q.kind == "choice" {
                "choice"
            } else {
                "score"
            },
            if q.kind == "choice" {
                DataType::Utf8
            } else {
                DataType::Float64
            },
            true,
        ),
        Field::new(
            "probabilities",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(p.into()),
                true,
            ))),
            true,
        ),
        Field::new("confidence", DataType::Float64, true),
    ]
    .into()
}
fn datatype(q: &Question) -> DataType {
    if q.kind == "noul" {
        DataType::Float64
    } else {
        DataType::Struct(fields(q))
    }
}
/// Text input, including dictionary-encoded text: low-cardinality columns are a
/// common shape for the categorical text callers feed to inference.
fn is_text(t: &DataType) -> bool {
    match t {
        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 | DataType::Null => true,
        DataType::Dictionary(_, inner) => matches!(
            inner.as_ref(),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
        ),
        _ => false,
    }
}
impl ScalarUDFImpl for Jev {
    fn name(&self) -> &str {
        "__datafusion_jev"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Err(plan_datafusion_err!(
            "prompt_jev requires constant configuration"
        ))
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        if args.arg_fields.len() != 2 || !is_text(args.arg_fields[0].data_type()) {
            return Err(plan_datafusion_err!("prompt_jev input must be text"));
        }
        let q =
            config(args.scalar_arguments[1].ok_or_else(|| {
                plan_datafusion_err!("prompt_jev configuration must be constant")
            })?)?;
        Ok(Arc::new(Field::new(self.name(), datatype(&q), true)))
    }
    fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(exec_datafusion_err!(
            "prompt_jev requires asynchronous execution"
        ))
    }
}
fn scalar_struct(fields: Fields, values: Vec<ScalarValue>) -> Result<ScalarValue> {
    let arrays: Result<Vec<ArrayRef>> = values.into_iter().map(|v| v.to_array()).collect();
    Ok(ScalarValue::Struct(Arc::new(StructArray::try_new(
        fields, arrays?, None,
    )?)))
}
fn number(value: &Value, upper: f64) -> Result<f64> {
    value
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0 && *v <= upper)
        .ok_or_else(|| exec_datafusion_err!("Jev returned an invalid numeric answer"))
}
fn answer(q: &Question, value: &Value) -> Result<ScalarValue> {
    if value["type"].as_str() != Some(q.kind.as_str()) {
        return Err(exec_datafusion_err!(
            "Jev returned an unexpected answer type"
        ));
    }
    if q.kind == "noul" {
        return Ok(ScalarValue::Float64(Some(number(&value["noul"], 1.0)?)));
    }
    let fs = fields(q);
    let DataType::List(item) = fs[1].data_type() else {
        unreachable!()
    };
    let DataType::Struct(pfields) = item.data_type() else {
        unreachable!()
    };
    let probs = value["probabilities"]
        .as_object()
        .ok_or_else(|| exec_datafusion_err!("Jev omitted probabilities"))?;
    if probs.len() != q.criteria.len() {
        return Err(exec_datafusion_err!(
            "Jev returned an unexpected probability count"
        ));
    }
    let mut entries = vec![];
    let mut sum = 0.0;
    for (i, c) in q.criteria.iter().enumerate() {
        let key = if q.kind == "score" {
            i.to_string()
        } else {
            c.label.clone()
        };
        let probability = number(probs.get(&key).unwrap_or(&Value::Null), 1.0)?;
        sum += probability;
        let mut values = vec![];
        if q.kind == "score" {
            values.push(ScalarValue::UInt32(Some(i as u32)));
        }
        values.push(ScalarValue::Utf8(Some(c.label.clone())));
        values.push(ScalarValue::Float64(Some(probability)));
        entries.push(scalar_struct(pfields.clone(), values)?);
    }
    // Providers round each probability independently, so the error a valid
    // answer can accumulate grows with the number of criteria.
    let tolerance = (0.01 + 0.001 * q.criteria.len() as f64).min(0.3);
    if (sum - 1.0).abs() > tolerance {
        return Err(exec_datafusion_err!("Jev probabilities do not sum to one"));
    }
    let decision = if q.kind == "choice" {
        let label = value["choice"]
            .as_str()
            .filter(|s| q.criteria.iter().any(|c| c.label == *s))
            .ok_or_else(|| exec_datafusion_err!("Jev returned an unknown choice"))?;
        ScalarValue::Utf8(Some(label.to_owned()))
    } else {
        ScalarValue::Float64(Some(number(
            &value["score"],
            (q.criteria.len() - 1) as f64,
        )?))
    };
    let probabilities = ScalarValue::List(ScalarValue::new_list(&entries, item.data_type(), true));
    scalar_struct(
        fs,
        vec![
            decision,
            probabilities,
            ScalarValue::Float64(Some(number(&value["confidence"], 1.0)?)),
        ],
    )
}
fn question_json(q: &Question) -> Value {
    let mut v = json!({"type":q.kind,"instructions":q.instructions});
    if !q.criteria.is_empty() {
        v["criteria"] = if q.kind == "score" {
            // Score criteria are an ordered array of strings; the provider has no
            // documented object form that carries a separate label. Keep the array
            // shape and fold the label into the text, so the caller's label still
            // reaches the model instead of being replaced by the description.
            Value::Array(
                q.criteria
                    .iter()
                    .map(|c| match &c.description {
                        Some(d) => json!(format!("{}: {d}", c.label)),
                        None => json!(c.label),
                    })
                    .collect(),
            )
        } else {
            Value::Object(
                q.criteria
                    .iter()
                    .map(|c| {
                        (
                            c.label.clone(),
                            json!(c.description.as_ref().unwrap_or(&c.label)),
                        )
                    })
                    .collect(),
            )
        };
    }
    v
}
fn request_body(q: &Question, rows: &[(usize, String)]) -> Value {
    if rows.len() == 1 {
        return json!({
            "model": "jev-latest",
            "state": rows[0].1,
            "questions": {"row_0": question_json(q)},
        });
    }
    let state: serde_json::Map<String, Value> = rows
        .iter()
        .enumerate()
        .map(|(i, (_, s))| (format!("row_{i}"), json!(s)))
        .collect();
    let questions: serde_json::Map<String, Value> = rows
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut question = question_json(q);
            question["instructions"] = json!({
                "question": q.instructions,
                "scope": format!(
                    "Evaluate only the text in state.row_{i}. Other rows are unrelated inputs."
                ),
            });
            (format!("row_{i}"), question)
        })
        .collect();
    json!({"model": "jev-latest", "state": state, "questions": questions})
}
impl Jev {
    async fn batch(
        &self,
        q: &Question,
        rows: Vec<(usize, String)>,
    ) -> Result<Vec<(usize, ScalarValue)>> {
        let body = request_body(q, &rows);
        // Backstop only: rows are already split by encoded size, so a single-row
        // batch cannot reach this limit through input length alone.
        if serde_json::to_vec(&body)
            .map_err(|_| exec_datafusion_err!("invalid Jev request"))?
            .len()
            > 256 * 1024
        {
            let hint = if rows.len() > 1 {
                "reduce batch_size or shorten the instructions"
            } else {
                "shorten the instructions"
            };
            return Err(exec_datafusion_err!("Jev request exceeds 256 KiB; {hint}"));
        }
        for attempt in 0..3 {
            // Hold a permit only while a request is in flight, never across backoff.
            let result = {
                let _permit = self
                    .permits
                    .acquire()
                    .await
                    .map_err(|_| exec_datafusion_err!("Jev inference closed"))?;
                tokio::time::timeout(Duration::from_secs(30), self.client.request(body.clone()))
                    .await
            };
            match result {
                Ok(Ok(response)) => {
                    let answers = response["answers"]
                        .as_object()
                        .ok_or_else(|| exec_datafusion_err!("Jev response omitted answers"))?;
                    return rows
                        .iter()
                        .enumerate()
                        .map(|(i, (row, _))| {
                            let v = answers.get(&format!("row_{i}")).ok_or_else(|| {
                                exec_datafusion_err!("Jev response omitted a row")
                            })?;
                            Ok((*row, answer(q, v)?))
                        })
                        .collect();
                }
                Ok(Err(RequestError::Fatal(message))) => {
                    return Err(exec_datafusion_err!("Jev request rejected: {message}"));
                }
                _ if attempt < 2 => {
                    tokio::time::sleep(Duration::from_millis(250 * (1 << attempt))).await
                }
                _ => {
                    return rows
                        .into_iter()
                        .map(|(i, _)| Ok((i, ScalarValue::try_from(&datatype(q))?)))
                        .collect();
                }
            }
        }
        unreachable!()
    }
}
#[async_trait]
impl AsyncScalarUDFImpl for Jev {
    fn ideal_batch_size(&self) -> Option<usize> {
        Some(256)
    }
    async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ColumnarValue::Scalar(config_value) = &args.args[1] else {
            return Err(exec_datafusion_err!(
                "prompt_jev configuration must be scalar"
            ));
        };
        let q = config(config_value)?;
        let len = args.number_rows;
        if len == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(&datatype(&q))));
        }
        let mut input = args.args[0].to_array(len)?;
        if matches!(input.data_type(), DataType::Dictionary(_, _)) {
            input = datafusion::arrow::compute::cast(&input, &DataType::Utf8)?;
        }
        // Ask about each distinct text once. Repeated and constant inputs are
        // common, and the answer is copied back to every row that shared the text.
        let mut unique: Vec<(String, usize)> = vec![];
        let mut rows_for: Vec<Vec<usize>> = vec![];
        let mut seen: HashMap<String, usize> = HashMap::new();
        for i in 0..len {
            if input.data_type() == &DataType::Null || input.is_null(i) {
                continue;
            }
            let s = match ScalarValue::try_from_array(&input, i)? {
                ScalarValue::Utf8(Some(s))
                | ScalarValue::Utf8View(Some(s))
                | ScalarValue::LargeUtf8(Some(s)) => s,
                _ => return Err(exec_datafusion_err!("prompt_jev input must be text")),
            };
            if let Some(&u) = seen.get(&s) {
                rows_for[u].push(i);
                continue;
            }
            // Measure what actually travels: JSON escaping can expand text severalfold.
            let encoded = serde_json::to_string(&s)
                .map_err(|_| exec_datafusion_err!("invalid Jev request"))?
                .len();
            if encoded > 64 * 1024 {
                return Err(exec_datafusion_err!(
                    "prompt_jev input exceeds 64 KiB once JSON-encoded"
                ));
            }
            seen.insert(s.clone(), unique.len());
            unique.push((s, encoded));
            rows_for.push(vec![i]);
        }
        let mut batches = vec![];
        let mut batch = vec![];
        let mut bytes = 0;
        for (u, (s, encoded)) in unique.into_iter().enumerate() {
            if !batch.is_empty() && (batch.len() >= q.batch_size || bytes + encoded > 32 * 1024) {
                batches.push(std::mem::take(&mut batch));
                bytes = 0;
            }
            bytes += encoded;
            batch.push((u, s));
        }
        if !batch.is_empty() {
            batches.push(batch);
        }
        let mut out = vec![ScalarValue::try_from(&datatype(&q))?; len];
        let mut pending =
            stream::iter(batches.into_iter().map(|rows| self.batch(&q, rows))).buffer_unordered(8);
        while let Some(result) = pending.next().await {
            for (u, value) in result? {
                for &row in &rows_for[u] {
                    out[row] = value.clone();
                }
            }
        }
        Ok(ColumnarValue::Array(ScalarValue::iter_to_array(out)?))
    }
}
