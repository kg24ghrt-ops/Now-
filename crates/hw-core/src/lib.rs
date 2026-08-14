//! Core orchestration for the handwriting engine.
//! Manages sessions, text processing, physics, and rendering.

use hw_ink::{build_ink_mesh, InkMesh};
use hw_paper::{PaperParams, PaperTexture};
use hw_physics::{HandProfile, PenSimulator, PointSample};
use hw_segment::{segment_text, Cluster};
use hw_shape::{GlyphSource, HarfBuzzGlyphSource, ShapedGlyph};
use kurbo::Point;
use std::collections::VecDeque;
use std::sync::Arc;

/// Main session handle. Holds all state for a single handwriting session.
pub struct Session {
    /// The text being rendered.
    text: String,
    /// Segmented clusters from the text.
    clusters: Vec<Cluster>,
    /// Glyph source (font + shaping).
    glyph_source: Arc<HarfBuzzGlyphSource>,
    /// Pen physics simulator.
    pen: PenSimulator,
    /// Generated ink mesh from the last render.
    mesh: Option<InkMesh>,
    /// Paper texture (generated once per session).
    paper_texture: PaperTexture,
    /// Current ink color (RGBA).
    ink_color: [f32; 4],
    /// Wetness (0..1) – controls ink bleed.
    wetness: f32,
    /// Base stroke width in pixels.
    base_width: f32,
    /// Paper texture scale for UV mapping.
    paper_scale: f32,
    /// Seed for deterministic RNG.
    seed: u64,
}

impl Session {
    /// Create a new session with a given font and seed.
    pub fn new(font_data: &[u8], seed: u64) -> Result<Self, hw_shape::ShapeError> {
        let glyph_source = Arc::new(HarfBuzzGlyphSource::from_bytes(font_data)?);
        let profile = HandProfile::default();
        let pen = PenSimulator::new(profile, seed);
        let paper_texture = PaperTexture::generate(&PaperParams::default());

        Ok(Self {
            text: String::new(),
            clusters: Vec::new(),
            glyph_source,
            pen,
            mesh: None,
            paper_texture,
            ink_color: [0.0, 0.0, 0.0, 1.0], // black ink
            wetness: 0.7,
            base_width: 2.5,
            paper_scale: 1.0,
            seed,
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
        self.clusters = segment_text(text);
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
            if let Some(path) = &glyph.outline {
                // Scale the path from font units to pixels.
                // We'll use a simple scale factor (adjust as needed).
                let scale = 0.1; // rough scaling – you'll want to calibrate this.
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

            // Add some random jitter for realism (the physics engine handles this).
        }

        if all_points.len() < 2 {
            self.mesh = None;
            return Ok(());
        }

        // Run the pen simulator over the path.
        let samples = self.pen.trace_path(&all_points, true);

        // Build the ink mesh from the samples.
        let mesh = build_ink_mesh(
            &samples,
            self.base_width,
            self.wetness,
            self.paper_scale,
        );

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