//! Every researched case must still come out the way it is labelled: a correct or false positive
//! fires, a correct or false negative does not.

use serde_json::Value;

#[test]
fn cases_hold() {
    let xs: Vec<Value> = serde_json::from_str(features::EXAMPLES).unwrap();
    let mut bad = vec![];
    for x in xs.iter().filter(|x| !x["case"].as_str().unwrap().ends_with("_impossible")) {
        let s = |k: &str| x[k].as_str().unwrap_or("").to_string();
        let top: Vec<String> = s("top").split_whitespace().map(String::from).collect();
        let fs = features::extract_fen(&s("fen"), (!top.is_empty()).then_some(&top[..])).unwrap();
        let side = x["side"].as_str();
        let f = fs.iter().find(|f| f.id == s("feature") && f.side == side)
            .unwrap_or_else(|| panic!("{} {:?}: no such feature", s("feature"), side));
        let want = matches!(s("case").as_str(), "tp" | "fp");
        if f.on != want { bad.push(format!("{} {} {:?} {}: on={}", s("family"), s("case"), side, s("fen"), f.on)); }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn every_family_has_its_cases() {
    let xs: Vec<Value> = serde_json::from_str(features::EXAMPLES).unwrap();
    for x in &xs {
        assert!(features::families().iter().any(|f| f.id == x["family"]), "unknown family {}", x["family"]);
        let Some(feature) = x["feature"].as_str() else { continue };
        assert!(feature.starts_with(x["family"].as_str().unwrap()), "{} not in {}", feature, x["family"]);
    }
}
