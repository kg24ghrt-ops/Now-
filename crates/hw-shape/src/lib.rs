use harfrust::{Face, Font, Features, Buffer, ShapeResult, GlyphInfo};
use read_fonts::TableProvider;
use kurbo::{BezPath, PathEl, Vec2};
use std::sync::Arc;

pub use harfrust::*;

/// Errors that can occur during shaping.
#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("Font loading failed: {0}")]
    FontLoad(String),
    #[error("Shaping failed for text")]
    ShapingFailed,
}

/// A source of glyph outlines – either vector (preferred) or bitmap fallback.
pub trait GlyphSource {
    /// Get the vector path for a given glyph ID, if available.
    /// The path is in font units; the caller must scale.
    fn get_outline(&self, glyph_id: u32) -> Option<BezPath>;
    /// Check if the glyph has a vector outline.
    fn has_vector_outline(&self, glyph_id: u32) -> bool;
    /// Get a bitmap (or fallback) representation; returns None if not supported.
    fn get_bitmap(&self, glyph_id: u32) -> Option<&[u8]>; // for now, just stub
    /// Get the advance width (in font units) for a glyph.
    fn get_advance(&self, glyph_id: u32) -> f64;
}

/// A HarfBuzz‑based glyph source that loads a font and shapes text.
pub struct HarfBuzzGlyphSource {
    font: Font,
    face: Face,
}

impl HarfBuzzGlyphSource {
    /// Load a font from raw font data (TTF/OTF).
    pub fn from_bytes(data: &[u8]) -> Result<Self, ShapeError> {
        let face = Face::from_bytes(data, 0)
            .map_err(|e| ShapeError::FontLoad(e.to_string()))?;
        let font = Font::new(face.clone(), Default::default());
        Ok(Self { font, face })
    }

    /// Shape a run of text, returning a list of glyphs with positions and outlines.
    pub fn shape_text(&self, text: &str) -> Result<Vec<ShapedGlyph>, ShapeError> {
        let mut buffer = Buffer::new();
        buffer.set_text(text);
        buffer.set_direction(harfrust::Direction::LeftToRight); // we can adjust later
        // You might want to set script/language from the segmenter.

        self.font.shape(&mut buffer, &Features::empty())
            .map_err(|_| ShapeError::ShapingFailed)?;

        let infos = buffer.glyph_infos();
        let positions = buffer.glyph_positions();
        let mut shaped = Vec::new();
        for (info, pos) in infos.iter().zip(positions.iter()) {
            let outline = self.get_outline(info.glyph);
            let advance = self.get_advance(info.glyph);
            shaped.push(ShapedGlyph {
                glyph_id: info.glyph,
                x_advance: pos.x_advance as f64,
                y_advance: pos.y_advance as f64,
                x_offset: pos.x_offset as f64,
                y_offset: pos.y_offset as f64,
                outline,
                advance_font_units: advance,
            });
        }
        Ok(shaped)
    }
}

impl GlyphSource for HarfBuzzGlyphSource {
    fn get_outline(&self, glyph_id: u32) -> Option<BezPath> {
        // Use read-fonts to extract outline.
        // The face has a `glyf` or `CFF` table. For simplicity, we'll extract via read-fonts.
        // We'll use the `outline` method from read-fonts if available.
        // This is a bit involved; we'll implement a helper.
        let glyph = self.face.glyph(glyph_id);
        // For TrueType, we can get the outline from the glyf table.
        // This returns an iterator of drawing commands.
        // We need to convert to kurbo BezPath.
        // We'll assume we have a helper function `outline_to_path`.
        outline_to_path(&self.face, glyph_id)
    }

    fn has_vector_outline(&self, glyph_id: u32) -> bool {
        // Check if the glyph has an outline (glyf or CFF).
        self.face.glyph(glyph_id).has_outline()
    }

    fn get_bitmap(&self, _glyph_id: u32) -> Option<&[u8]> {
        // Not implemented yet – will later use embedded bitmap or rasterized fallback.
        None
    }

    fn get_advance(&self, glyph_id: u32) -> f64 {
        // In font units.
        self.face.glyph(glyph_id).advance().unwrap_or(0) as f64
    }
}

/// A shaped glyph with its outline and placement info.
#[derive(Debug, Clone)]
pub struct ShapedGlyph {
    pub glyph_id: u32,
    pub x_advance: f64,
    pub y_advance: f64,
    pub x_offset: f64,
    pub y_offset: f64,
    pub outline: Option<BezPath>,
    pub advance_font_units: f64,
}

/// Helper to convert a read-fonts outline to a kurbo BezPath.
fn outline_to_path(face: &read_fonts::Face, glyph_id: u32) -> Option<BezPath> {
    use read_fonts::types::Point;
    use read_fonts::tables::glyf::Glyph;
    use read_fonts::tables::cff::Cff;

    // We'll try both glyf and CFF.
    // For TrueType glyf:
    if let Ok(Some(glyf_table)) = face.glyf() {
        if let Some(glyph) = glyf_table.glyph(glyph_id) {
            let mut path = BezPath::new();
            // Iterate over the contours.
            // This is a simplified version; we need to handle on-curve/off-curve.
            // We'll implement a proper converter later.
            // For now, return a simple placeholder.
            // (In production, you'd iterate over the contour's points and build the path.)
            // We'll just return None to indicate not yet fully implemented.
            return None;
        }
    }
    // Try CFF if available.
    if let Ok(Some(cff_table)) = face.cff() {
        // ...
    }
    None
}