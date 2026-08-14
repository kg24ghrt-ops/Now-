//! Myanmar (Burmese) script pack for the handwriting engine.
//! Handles complex cluster segmentation, fallback detection, and contextual shaping.

use hw_core::ScriptPolicy;
use hw_segment::{Cluster, Script};
use hw_shape::{GlyphProbe, ShapedRun};
use kurbo::BezPath;
use unicode_segmentation::UnicodeSegmentation;

/// Myanmar script policy.
pub struct MyanmarPolicy;

impl MyanmarPolicy {
    /// Check if a codepoint is a Myanmar character.
    fn is_myanmar_char(ch: char) -> bool {
        matches!(ch,
            '\u{1000}'..='\u{109F}' // Myanmar block
            | '\u{AA60}'..='\u{AA7F}' // Myanmar Extended-A
            | '\u{A9E0}'..='\u{A9FF}' // Myanmar Extended-B
        )
    }

    /// Check if a character is a vowel or tone mark (dependent sign).
    fn is_dependent_sign(ch: char) -> bool {
        matches!(ch,
            '\u{102B}'..='\u{1030}' // ါ ိ ီ ု ူ ေ ဲ
            | '\u{1032}'..='\u{1037}' // ဳ ဴ ဵ ံ ့ း
            | '\u{103C}' // ြ
            | '\u{103D}' // ွ
            | '\u{103E}' // ှ
            | '\u{103F}' // ဿ
            // Tone marks
            | '\u{103A}' // ယ
            | '\u{103B}' // ရ
        )
    }

    /// Check if a character is a medial (consonant modifier).
    fn is_medial(ch: char) -> bool {
        matches!(
            ch,
            '\u{103C}' // ြ (ya-pin)
            | '\u{103D}' // ွ (wa-pin)
            | '\u{103E}' // ှ (ha-pin)
        )
    }

    /// Check if a character is a consonant (including stacked).
    fn is_consonant(ch: char) -> bool {
        matches!(
            ch,
            '\u{1000}'
                ..='\u{102A}' // consonants and some vowels
            | '\u{103B}' // ရ (ra)
        )
    }

    /// Custom cluster joining for Myanmar: combine consonants with following medial/vowel/tone marks.
    fn join_myanmar_clusters<'a>(&self, clusters: &[&'a str]) -> Vec<Cluster> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < clusters.len() {
            let mut combined = String::new();
            let mut script = Script::Unknown;

            // Start with the base cluster.
            let base = clusters[i];
            combined.push_str(base);
            if let Some(ch) = base.chars().next() {
                script = ch.script();
            }

            // Look ahead: if we have a consonant followed by dependent signs,
            // combine them into one cluster.
            let mut j = i + 1;
            while j < clusters.len() {
                let next = clusters[j];
                if let Some(ch) = next.chars().next() {
                    if Self::is_dependent_sign(ch) || Self::is_medial(ch) {
                        combined.push_str(next);
                        j += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            // Also handle stacked consonants (e.g., မ္ဘ) – simplified.
            // For now, we keep them separate.

            result.push(Cluster {
                text: combined,
                script,
            });

            i = j;
        }

        result
    }
}

impl ScriptPolicy for MyanmarPolicy {
    fn cluster_join_rule(&self, clusters: &[&str]) -> Vec<Cluster> {
        self.join_myanmar_clusters(clusters)
    }

    fn requires_bitmap_fallback(&self, glyph: &GlyphProbe) -> bool {
        if let Some(outline) = glyph.vector_outline() {
            // Check if the outline has actual geometry.
            let has_geometry = outline.elements().iter().any(|el| {
                matches!(
                    el,
                    kurbo::PathEl::LineTo(_)
                        | kurbo::PathEl::QuadTo(_, _)
                        | kurbo::PathEl::CurveTo(_, _, _)
                )
            });
            !has_geometry
        } else {
            true
        }
    }

    fn contextual_shaping(&self, run: &str) -> ShapedRun {
        // Default shaping; HarfBuzz handles Myanmar.
        ShapedRun::new(run)
    }
}

impl Default for MyanmarPolicy {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_joining() {
        let policy = MyanmarPolicy;
        let graphemes: Vec<&str> = UnicodeSegmentation::graphemes("မင်္ဂလာ", true).collect();
        let clusters = policy.cluster_join_rule(&graphemes);
        assert!(clusters.len() < graphemes.len());
        assert!(clusters[0].text.contains("မ"));
        assert!(clusters[0].text.contains("င်"));
    }
}
