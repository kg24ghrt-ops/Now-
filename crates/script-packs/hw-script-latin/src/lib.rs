//! Latin script pack – simplest case.
//! Uses default Unicode segmentation with no special rules.

use hw_core::ScriptPolicy;
use hw_segment::{Cluster, Script};
use hw_shape::{GlyphProbe, ShapedRun};

/// Latin script policy – the simplest case.
pub struct LatinPolicy;

impl ScriptPolicy for LatinPolicy {
    fn cluster_join_rule(&self, clusters: &[&str]) -> Vec<Cluster> {
        clusters
            .iter()
            .map(|s| {
                let script = if let Some(ch) = s.chars().next() {
                    ch.script()
                } else {
                    Script::Unknown
                };
                Cluster {
                    text: s.to_string(),
                    script,
                }
            })
            .collect()
    }

    fn requires_bitmap_fallback(&self, glyph: &GlyphProbe) -> bool {
        glyph.vector_outline().is_none()
    }

    fn contextual_shaping(&self, run: &str) -> ShapedRun {
        ShapedRun::new(run)
    }
}

impl Default for LatinPolicy {
    fn default() -> Self {
        Self
    }
}