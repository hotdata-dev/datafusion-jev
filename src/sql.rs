//! Lower the public SQL syntax to a validated, constant configuration.
use datafusion::{
    common::{Result, plan_datafusion_err},
    sql::sqlparser::ast::{
        self, Expr, FunctionArg, FunctionArgExpr, FunctionArguments, Ident, ObjectName, Value,
    },
};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, ops::ControlFlow};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Criterion {
    pub label: String,
    pub description: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Question {
    pub instructions: String,
    pub kind: String,
    pub criteria: Vec<Criterion>,
    pub batch_size: usize,
}
impl Question {
    pub fn validate(&self) -> Result<()> {
        if self.instructions.trim().is_empty() {
            return Err(plan_datafusion_err!(
                "prompt_jev instructions must not be empty"
            ));
        }
        if !(1..=64).contains(&self.batch_size) {
            return Err(plan_datafusion_err!(
                "prompt_jev batch_size must be between 1 and 64"
            ));
        }
        let mut labels = HashSet::new();
        for c in &self.criteria {
            if c.label.trim().is_empty()
                || !labels.insert(&c.label)
                || c.description.as_ref().is_some_and(|s| s.trim().is_empty())
            {
                return Err(plan_datafusion_err!(
                    "prompt_jev criteria require unique non-empty labels and non-empty descriptions"
                ));
            }
        }
        let valid = match self.kind.as_str() {
            "choice" => (2..=255).contains(&self.criteria.len()),
            "score" => (2..=10).contains(&self.criteria.len()),
            "noul" => {
                self.criteria.is_empty()
                    || (labels.len() == 2
                        && labels.contains(&"true".to_owned())
                        && labels.contains(&"false".to_owned()))
            }
            _ => false,
        };
        if !valid {
            return Err(plan_datafusion_err!(
                "prompt_jev invalid criteria for {}",
                self.kind
            ));
        }
        Ok(())
    }
}
fn string(e: &Expr) -> Result<String> {
    match e {
        Expr::Value(v) => match &v.value {
            Value::SingleQuotedString(s) | Value::EscapedStringLiteral(s) => Ok(s.clone()),
            _ => Err(plan_datafusion_err!("prompt_jev expects a constant string")),
        },
        _ => Err(plan_datafusion_err!("prompt_jev expects a constant string")),
    }
}
fn criteria(e: &Expr) -> Result<Vec<Criterion>> {
    let Expr::Array(a) = e else {
        return Err(plan_datafusion_err!(
            "prompt_jev criteria must be a constant array"
        ));
    };
    a.elem
        .iter()
        .map(|e| {
            let Expr::Dictionary(fields) = e else {
                return Ok(Criterion {
                    label: string(e)?,
                    description: None,
                });
            };
            let mut label = None;
            let mut description = None;
            let mut seen = HashSet::new();
            for field in fields {
                let key = field.key.value.as_str();
                if !seen.insert(key) {
                    return Err(plan_datafusion_err!("duplicate criterion field"));
                }
                match key {
                    "label" => label = Some(string(&field.value)?),
                    "description" => {
                        let null = matches!(
                            field.value.as_ref(),
                            Expr::Value(v) if v.value == Value::Null
                        );
                        if !null {
                            description = Some(string(&field.value)?);
                        }
                    }
                    _ => return Err(plan_datafusion_err!("unknown criterion field: {key}")),
                }
            }
            Ok(Criterion {
                label: label.ok_or_else(|| plan_datafusion_err!("criterion requires label"))?,
                description,
            })
        })
        .collect()
}
/// Call after parsing and before DataFusion plans the statement. Idempotent.
pub fn rewrite_statement(statement: &mut ast::Statement) -> Result<()> {
    let flow = ast::visit_expressions_mut(statement, |expr| {
        let Expr::Function(f) = expr else {
            return ControlFlow::Continue(());
        };
        if f.name.to_string().to_lowercase() != "prompt_jev" {
            return ControlFlow::Continue(());
        }
        let result = (|| {
            if f.filter.is_some()
                || f.over.is_some()
                || f.null_treatment.is_some()
                || !f.within_group.is_empty()
                || !matches!(f.parameters, FunctionArguments::None)
            {
                return Err(plan_datafusion_err!(
                    "prompt_jev does not accept aggregate/window modifiers"
                ));
            }
            let FunctionArguments::List(args) = &mut f.args else {
                return Err(plan_datafusion_err!(
                    "prompt_jev requires input and instructions"
                ));
            };
            if args.duplicate_treatment.is_some() || !args.clauses.is_empty() {
                return Err(plan_datafusion_err!("invalid prompt_jev modifiers"));
            }
            let mut positional = Vec::new();
            let mut named = HashSet::new();
            let mut q = Question {
                instructions: String::new(),
                kind: "noul".into(),
                criteria: vec![],
                batch_size: 32,
            };
            let mut mode = false;
            for arg in &args.args {
                match arg {
                    FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) if named.is_empty() => {
                        positional.push(e.clone())
                    }
                    FunctionArg::Named {
                        name,
                        arg: FunctionArgExpr::Expr(e),
                        ..
                    } => {
                        let name = if name.quote_style.is_some() {
                            name.value.clone()
                        } else {
                            name.value.to_lowercase()
                        };
                        if !named.insert(name.clone()) {
                            return Err(plan_datafusion_err!(
                                "duplicate prompt_jev argument: {name}"
                            ));
                        }
                        match name.as_str() {
                            "choice" | "score" | "noul" => {
                                if mode {
                                    return Err(plan_datafusion_err!(
                                        "choice, score, and noul are mutually exclusive"
                                    ));
                                }
                                mode = true;
                                q.kind = name;
                                q.criteria = criteria(e)?;
                                if q.kind == "noul" && q.criteria.is_empty() {
                                    return Err(plan_datafusion_err!(
                                        "explicit noul criteria require true and false"
                                    ));
                                }
                            }
                            "batch_size" => {
                                q.batch_size = match e {
                                    Expr::Value(v) => match &v.value {
                                        Value::Number(n, _) => n.parse().ok(),
                                        _ => None,
                                    },
                                    _ => None,
                                }
                                .ok_or_else(|| {
                                    plan_datafusion_err!("batch_size must be a constant integer")
                                })?;
                            }
                            _ => {
                                return Err(plan_datafusion_err!(
                                    "unsupported prompt_jev argument: {name}"
                                ));
                            }
                        }
                    }
                    _ => {
                        return Err(plan_datafusion_err!(
                            "prompt_jev requires two positional arguments followed by named options"
                        ));
                    }
                }
            }
            if positional.len() != 2 {
                return Err(plan_datafusion_err!(
                    "prompt_jev requires input and constant instructions"
                ));
            }
            q.instructions = string(&positional[1])?;
            q.validate()?;
            let config = serde_json::to_string(&q).map_err(|e| plan_datafusion_err!("{e}"))?;
            f.name = ObjectName::from(vec![Ident::new("__datafusion_jev")]);
            args.args = vec![
                FunctionArg::Unnamed(FunctionArgExpr::Expr(positional.remove(0))),
                FunctionArg::Unnamed(FunctionArgExpr::Expr(Expr::Value(
                    Value::SingleQuotedString(config).into(),
                ))),
            ];
            Ok(())
        })();
        match result {
            Ok(()) => ControlFlow::Continue(()),
            Err(e) => ControlFlow::Break(e),
        }
    });
    match flow {
        ControlFlow::Continue(()) => Ok(()),
        ControlFlow::Break(e) => Err(e),
    }
}
