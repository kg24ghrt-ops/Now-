/data/data/org.smartide.code/files/home/projects/fc/crates/hw-core/src/lib.rs:

//! Core orchestration for the handwriting engine.
//! Manages sessions, text processing, physics, and rendering.

use hw_core::ScriptPolicy;
use hw_ink::{build_ink_mesh, InkMesh};
use hw_paper::{PaperParams, PaperTexture};
use hw_physics::{HandProfile, PenSimulator, PointSample};
use hw_segment::{detect_script_run, segment_text, Cluster, Script};
use hw_shape::{GlyphSource, HarfBuzzGlyphSource, ShapedGlyph};
// Import the script packs.
use hw_script_arabic::ArabicPolicy;
use hw_script_latin::LatinPolicy;
use hw_script_myanmar::MyanmarPolicy;
use kurbo::Point;
use std::collections::VecDeque;
use std::sync::Arc;

/// Main session handle. Holds all state for a single handwriting session.
pub struct Session {
    text: String,
    clusters: Vec<Cluster>,
    glyph_source: Arc<HarfBuzzGlyphSource>,
    pen: PenSimulator,
    mesh: Option<InkMesh>,
    paper_texture: PaperTexture,
    ink_color: [f32; 4],
    wetness: f32,
    base_width: f32,
    paper_scale: f32,
    seed: u64,
    script_policy: Box<dyn ScriptPolicy>,
}

impl Session {
    /// Create a new session with a given font and seed.
    /// The script is auto‑detected from the text later.
    pub fn new(font_data: &[u8], seed: u64) -> Result<Self, hw_shape::ShapeError> {
        let glyph_source = Arc::new(HarfBuzzGlyphSource::from_bytes(font_data)?);
        let profile = HandProfile::default();
        let pen = PenSimulator::new(profile, seed);
        let paper_texture = PaperTexture::generate(&PaperParams::default());

        // Default to Latin until we have text.
        let script_policy: Box<dyn ScriptPolicy> = Box::new(LatinPolicy::default());

        Ok(Self {
            text: String::new(),
            clusters: Vec::new(),
            glyph_source,
            pen,
            mesh: None,
            paper_texture,
            ink_color: [0.0, 0.0, 0.0, 1.0],
            wetness: 0.7,
            base_width: 2.5,
            paper_scale: 1.0,
            seed,
            script_policy,
        })
    }

    /// Set the ink color (RGBA, 0..1).
    pub fn set_ink_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.ink_color = [r, g, b, a];
    }

    /// Set wetness (0 = no bleed, 1 = maximum bleed).
    pub fn set_wetness(&mut self, wetness: f32) {
        self.wetness = wetness.clamp(0.0, 1.0);
    }

    /// Set the base stroke width in pixels.
    pub fn set_base_width(&mut self, width: f32) {
        self.base_width = width.max(0.5);
    }

    /// Feed new text into the session. This replaces any previous text.
    pub fn feed_text(&mut self, text: &str) {
        self.text = text.to_string();

        // Auto‑detect the script of the entire text.
        let script = detect_script_run(text);
        // Select the appropriate policy.
        self.script_policy = match script {
            Script::Myanmar => Box::new(MyanmarPolicy::default()),
            Script::Arabic | Script::Syriac | Script::Thaana => Box::new(ArabicPolicy::default()),
            // Add other scripts as needed.
            _ => Box::new(LatinPolicy::default()),
        };

        // Segment using the policy's cluster join rule.
        let raw_clusters = segment_text(text);
        let cluster_strings: Vec<&str> = raw_clusters.iter().map(|c| c.text.as_str()).collect();
        self.clusters = self.script_policy.cluster_join_rule(&cluster_strings);

        // Reset pen position for the new text.
        self.pen.reset();
        self.mesh = None;
    }

    /// Process the current text through the full pipeline and generate an ink mesh.
    pub fn process(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.text.is_empty() {
            self.mesh = None;
            return Ok(());
        }

        // Shape the text into glyphs.
        let shaped = self.glyph_source.shape_text(&self.text)?;

        // Convert shaped glyphs into a path of points.
        let mut all_points: Vec<Point> = Vec::new();
        let mut current_x = 0.0;
        let mut current_y = 0.0;

        // We'll simulate writing each glyph.
        for glyph in &shaped {
            // Get the outline path for this glyph.
            // Check if we need fallback via the policy.
            let probe = hw_shape::GlyphProbe::from_glyph_info(glyph);
            if self.script_policy.requires_bitmap_fallback(&probe) {
                // For now, we just skip (or we could use a bitmap slice fallback later).
                current_x += glyph.x_advance * 0.1;
                continue;
            }

            if let Some(path) = &glyph.outline {
                // Scale the path from font units to pixels.
                let scale = 0.1; // rough scaling – adjust as needed.
                let mut points: Vec<Point> = path
                    .elements()
                    .iter()
                    .filter_map(|el| match el {
                        kurbo::PathEl::MoveTo(p) => Some(*p),
                        kurbo::PathEl::LineTo(p) => Some(*p),
                        kurbo::PathEl::QuadTo(_, p) => Some(*p),
                        kurbo::PathEl::CurveTo(_, _, p) => Some(*p),
                        kurbo::PathEl::ClosePath => None,
                    })
                    .map(|p| Point::new(p.x * scale + current_x, p.y * scale + current_y))
                    .collect();

                all_points.append(&mut points);

                // Advance the pen position by the glyph's advance.
                current_x += glyph.x_advance * scale;
                // Add some spacing bias (from the physics profile).
                current_x += self.pen.profile().spacing_bias.bias_per_char as f64 * 0.1;
            } else {
                // Fallback: if no outline, just advance.
                current_x += glyph.x_advance * 0.1;
            }
        }

        if all_points.len() < 2 {
            self.mesh = None;
            return Ok(());
        }

        // Run the pen simulator over the path.
        let samples = self.pen.trace_path(&all_points, true);

        // Build the ink mesh from the samples.
        let mesh = build_ink_mesh(&samples, self.base_width, self.wetness, self.paper_scale);

        self.mesh = Some(mesh);
        Ok(())
    }

    /// Get the generated ink mesh (if any).
    pub fn mesh(&self) -> Option<&InkMesh> {
        self.mesh.as_ref()
    }

    /// Get the paper texture.
    pub fn paper_texture(&self) -> &PaperTexture {
        &self.paper_texture
    }

    /// Get the current ink color.
    pub fn ink_color(&self) -> [f32; 4] {
        self.ink_color
    }

    /// Get the current wetness.
    pub fn wetness(&self) -> f32 {
        self.wetness
    }

    /// Get a mutable reference to the pen simulator (for advanced tuning).
    pub fn pen_mut(&mut self) -> &mut PenSimulator {
        &mut self.pen
    }

    /// Get the current text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get the clusters.
    pub fn clusters(&self) -> &[Cluster] {
        &self.clusters
    }
}
