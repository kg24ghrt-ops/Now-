use unicode_script::{Script, UnicodeScript};
use unicode_segmentation::UnicodeSegmentation;

/// A cluster is a sequence of codepoints that form a single grapheme cluster,
/// plus its script property (derived from the first codepoint, or "Unknown" if empty).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    pub text: String,
    pub script: Script,
}

/// Split the input text into grapheme clusters and detect script for each.
/// Returns a vector of clusters.
pub fn segment_text(text: &str) -> Vec<Cluster> {
    UnicodeSegmentation::graphemes(text, true)
        .map(|g| {
            let script = if g.is_empty() {
                Script::Unknown
            } else {
                // Use the first char's script; in practice, most clusters are homogeneous.
                let first = g.chars().next().unwrap();
                first.script()
            };
            Cluster {
                text: g.to_string(),
                script,
            }
        })
        .collect()
}

/// Determine the overall script of a run of text by majority vote.
/// Used to pick a ScriptPolicy.
pub fn detect_script_run(text: &str) -> Script {
    let clusters = segment_text(text);
    let mut counts = std::collections::HashMap::new();
    for cl in &clusters {
        if cl.script != Script::Unknown {
            *counts.entry(cl.script).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(script, _)| script)
        .unwrap_or(Script::Unknown)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_myanmar() {
        let text = "မင်္ဂလာပါ";
        let clusters = segment_text(text);
        assert_eq!(clusters.len(), 5); // Myanmar graphemes
        for cl in &clusters {
            assert_eq!(cl.script, Script::Myanmar);
        }
    }
}
