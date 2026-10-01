#![cfg_attr(not(any(test, target_arch = "wasm32")), allow(dead_code))]

//! The DSL as text: the `Debug` form without quotes, e.g.
//! `Because(Rated(Blunder), Allows([Qh5+, g6, Qxg6+], Mate))`.

use crate::dsl_claude::Reason;

enum Node {
    Atom(String),
    Call(String, Vec<Node>),
    Fields(String, Vec<(String, Node)>),
    List(Vec<Node>),
    Tuple(Vec<Node>),
}

const PUNCT: &str = "()[]{},:";

fn tokens(src: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for c in src.chars() {
        if c.is_whitespace() || PUNCT.contains(c) {
            if !cur.is_empty() { out.push(std::mem::take(&mut cur)); }
            if !c.is_whitespace() { out.push(c.to_string()); }
        } else {
            cur.push(c);
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

struct P { t: Vec<String>, i: usize }

impl P {
    fn peek(&self) -> Option<&str> { self.t.get(self.i).map(|s| s.as_str()) }
    fn next(&mut self) -> Result<String, String> {
        let t = self.t.get(self.i).cloned().ok_or("unexpected end")?;
        self.i += 1;
        Ok(t)
    }
    fn expect(&mut self, s: &str) -> Result<(), String> {
        let t = self.next()?;
        if t == s { Ok(()) } else { Err(format!("expected `{s}`, found `{t}`")) }
    }
    /// Comma-separated items up to `close`; a trailing comma is allowed.
    fn items<T>(&mut self, close: &str, mut item: impl FnMut(&mut P) -> Result<T, String>) -> Result<Vec<T>, String> {
        let mut xs = vec![];
        loop {
            if self.peek() == Some(close) { self.i += 1; return Ok(xs); }
            xs.push(item(self)?);
            match self.next()?.as_str() {
                "," => {}
                t if t == close => return Ok(xs),
                t => return Err(format!("expected `,` or `{close}`, found `{t}`")),
            }
        }
    }
    fn node(&mut self) -> Result<Node, String> {
        let t = self.next()?;
        match t.as_str() {
            "[" => Ok(Node::List(self.items("]", P::node)?)),
            "(" => Ok(Node::Tuple(self.items(")", P::node)?)),
            t if PUNCT.contains(t) => Err(format!("unexpected `{t}`")),
            _ => match self.peek() {
                Some("(") => { self.i += 1; Ok(Node::Call(t, self.items(")", P::node)?)) }
                Some("{") => {
                    self.i += 1;
                    let fs = self.items("}", |p| { let k = p.next()?; p.expect(":")?; Ok((k, p.node()?)) })?;
                    Ok(Node::Fields(t, fs))
                }
                _ => Ok(Node::Atom(t)),
            },
        }
    }
}

fn tree(src: &str) -> Result<Node, String> {
    let mut p = P { t: tokens(src), i: 0 };
    let n = p.node()?;
    match p.peek() {
        None => Ok(n),
        Some(t) => Err(format!("unexpected `{t}` after the reason")),
    }
}

/// serde's externally tagged JSON: `Name(a)` → `{"Name": a}`, `Name(a, b)` → `{"Name": [a, b]}`.
fn json(n: &Node) -> Result<String, String> {
    let seq = |xs: &[Node]| -> Result<String, String> { Ok(format!("[{}]", xs.iter().map(json).collect::<Result<Vec<_>, _>>()?.join(","))) };
    Ok(match n {
        Node::Atom(a) if a == "None" => "null".into(),
        Node::Atom(a) if a == "true" || a == "false" || a.parse::<i64>().is_ok() => a.clone(),
        Node::Atom(a) if a.contains(['"', '\\']) => return Err(format!("bad token `{a}`")),
        Node::Atom(a) => format!("\"{a}\""),
        Node::Call(f, xs) if f == "Some" && xs.len() == 1 => json(&xs[0])?,
        Node::Call(f, xs) if xs.len() == 1 => format!("{{\"{f}\":{}}}", json(&xs[0])?),
        Node::Call(f, xs) => format!("{{\"{f}\":{}}}", seq(xs)?),
        Node::Fields(f, fs) => {
            let body = fs.iter().map(|(k, v)| Ok(format!("\"{k}\":{}", json(v)?))).collect::<Result<Vec<_>, String>>()?;
            format!("{{\"{f}\":{{{}}}}}", body.join(","))
        }
        Node::List(xs) | Node::Tuple(xs) => seq(xs)?,
    })
}

/// Parses DSL text. The text is leaked: the DSL holds `&'static str`.
pub(crate) fn parse(src: &str) -> Result<Reason, String> {
    let j: &'static str = Box::leak(json(&tree(src)?)?.into_boxed_str());
    serde_json::from_str(j).map_err(|e| {
        let e = e.to_string();
        // serde lists every variant after "expected one of"; the name alone is enough.
        e.split(", expected").next().unwrap_or(&e).split(" at line").next().unwrap_or(&e).to_string()
    })
}

fn flat(n: &Node) -> String {
    let join = |xs: &[Node]| xs.iter().map(flat).collect::<Vec<_>>().join(", ");
    match n {
        Node::Atom(a) => a.clone(),
        Node::Call(f, xs) => format!("{f}({})", join(xs)),
        Node::Fields(f, fs) => format!("{f} {{ {} }}", fs.iter().map(|(k, v)| format!("{k}: {}", flat(v))).collect::<Vec<_>>().join(", ")),
        Node::List(xs) => format!("[{}]", join(xs)),
        Node::Tuple(xs) => format!("({})", join(xs)),
    }
}

const WIDTH: usize = 64;

fn pretty(n: &Node, ind: usize) -> String {
    let f = flat(n);
    if ind + f.len() <= WIDTH { return f; }
    let pad = " ".repeat(ind + 2);
    let block = |open: String, xs: Vec<String>, close: &str| format!("{open}\n{pad}{}\n{}{close}", xs.join(&format!(",\n{pad}")), " ".repeat(ind));
    match n {
        Node::Atom(a) => a.clone(),
        // `And([` ... `])`: a lone list argument shares the call's brackets and indent.
        Node::Call(f, xs) if xs.len() == 1 && matches!(xs[0], Node::List(_)) => {
            let Node::List(ys) = &xs[0] else { unreachable!() };
            block(format!("{f}(["), ys.iter().map(|y| pretty(y, ind + 2)).collect(), "])")
        }
        Node::Call(f, xs) => block(format!("{f}("), xs.iter().map(|x| pretty(x, ind + 2)).collect(), ")"),
        Node::Fields(f, fs) => block(format!("{f} {{"), fs.iter().map(|(k, v)| format!("{k}: {}", pretty(v, ind + 4 + k.len()))).collect(), "}"),
        Node::List(xs) => block("[".into(), xs.iter().map(|x| pretty(x, ind + 2)).collect(), "]"),
        Node::Tuple(xs) => block("(".into(), xs.iter().map(|x| pretty(x, ind + 2)).collect(), ")"),
    }
}

/// The reason as DSL text, broken over lines where it is long.
pub(crate) fn show(r: &Reason) -> String {
    let d = format!("{r:?}").replace(['"', '\''], "");
    tree(&d).map(|n| pretty(&n, 0)).unwrap_or(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl_claude::examples;

    #[test]
    fn every_example_round_trips() {
        for ex in examples() {
            let s = show(&ex.reason);
            let back = parse(&s).unwrap_or_else(|e| panic!("{e}\n{s}"));
            assert_eq!(format!("{:?}", ex.reason), format!("{back:?}"), "\n{s}");
        }
    }

    #[test]
    fn errors_are_short() {
        let e = parse("And([Mate, Frobnicate])").unwrap_err();
        assert!(e.contains("Frobnicate") && !e.contains("expected one of"), "{e}");
        assert!(parse("Mate)").is_err());
        assert!(parse("Allows([Qh5+], ").is_err());
    }
}
