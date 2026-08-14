//! Converts simulated pen samples into GPU‑tessellated ink meshes.

use bytemuck::{Pod, Zeroable};
use hw_physics::PointSample;
use kurbo::{BezPath, PathEl, Point, Vec2};
use lyon::math::Point as LyonPoint;
use lyon::path::builder::*;
use lyon::path::Path as LyonPath;
use lyon::tessellation::{FillTessellator, StrokeOptions, StrokeTessellator, VertexBuffers};
use std::f32::consts::PI;

/// A single vertex for the ink mesh.
/// This layout matches the WGSL shader expected by `hw-render-wgpu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InkVertex {
    pub position: [f32; 3], // x, y, z (z is layer depth, unused for now)
    pub uv: [f32; 2],       // paper texture coordinates (will be set later)
    pub pressure: f32,      // 0..1
    pub wetness: f32,       // 0..1, controls ink diffusion
    pub normal: [f32; 2],   // for lighting effects
}

/// A complete ink mesh ready for GPU upload.
#[derive(Debug, Clone)]
pub struct InkMesh {
    pub vertices: Vec<InkVertex>,
    pub indices: Vec<u32>,
}

impl InkMesh {
    /// Returns the total number of triangles.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Builds an ink mesh from a list of point samples.
/// Uses `lyon` to tessellate a stroke with variable width.
pub fn build_ink_mesh(
    samples: &[PointSample],
    base_width: f32,
    wetness: f32,
    paper_texture_scale: f32,
) -> InkMesh {
    if samples.len() < 2 {
        return InkMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    // Convert samples to a kurbo path (interpolated with bezier curves).
    let path = build_kurbo_path(samples);

    // Convert kurbo path to lyon path.
    let lyon_path = convert_kurbo_to_lyon(&path);

    // Tessellate with variable stroke width.
    let mut buffers: VertexBuffers<InkVertex, u32> = VertexBuffers::new();

    // We'll use a stroke tessellator.
    let mut tessellator = StrokeTessellator::new();

    // Options: no miter limit (we want round joins), use tolerance.
    let options = StrokeOptions::default()
        .with_line_width(1.0) // base width, we'll adjust per vertex.
        .with_tolerance(0.1)
        .with_line_join(lyon::tessellation::LineJoin::Round)
        .with_line_cap(lyon::tessellation::LineCap::Round);

    // We need to add vertices with varying width.
    // The `tessellate_with_vertices` method accepts a closure that provides
    // per‑vertex attributes. We'll build an array of widths.
    // But a simpler approach: since `lyon` 1.0 doesn't directly support varying
    // width in the `StrokeTessellator` API without custom vertex construction,
    // we can use the FillTessellator on a generated outline.
    // Alternatively, we can use the `StrokeTessellator` with a `StrokeVertex` iterator.
    // The recommended way: collect path vertices with attributes.

    // For simplicity and robustness, we'll use the `StrokeTessellator` with a
    // `StrokeVertex` builder. We'll implement a custom iterator that yields
    // `StrokeVertex` with varying width.
    // `lyon` provides `StrokeVertex` in the `lyon::tessellation::StrokeVertex` type.

    // We'll need to create a path that includes the width information.
    // Because `lyon`'s stroke tessellator expects a constant width if we use
    // the simple `tessellate_path`, we'll instead build the stroke geometry manually
    // by creating offset curves. But that's complex.

    // Better: Use `lyon`'s `FillTessellator` on a thickened path.
    // We'll generate a polygon outline by expanding the stroke with variable width.
    // Since we have dense samples, we can compute the left and right offset points
    // for each sample, and then tessellate the resulting ribbon.

    // Let's implement a manual ribbon tessellation because it gives us full
    // control over per‑vertex attributes (pressure, wetness).

    // --- Manual ribbon tessellation ---
    let mut verts = Vec::new();
    let mut idxs = Vec::new();

    let half_width = base_width * 0.5;

    // For each sample, compute normal direction and offset.
    let mut last_pos = samples[0].position;
    let mut last_normal = Vec2::ZERO;

    // Pre‑compute normals for each point.
    let mut normals = Vec::with_capacity(samples.len());
    for i in 0..samples.len() {
        let p = samples[i].position;
        let mut normal = Vec2::ZERO;
        if i == 0 {
            let dir = samples[1].position - p;
            normal = Vec2::new(-dir.y, dir.x).normalize();
        } else if i == samples.len() - 1 {
            let dir = p - samples[i - 1].position;
            normal = Vec2::new(-dir.y, dir.x).normalize();
        } else {
            let dir1 = p - samples[i - 1].position;
            let dir2 = samples[i + 1].position - p;
            let dir = (dir1 + dir2) * 0.5;
            normal = Vec2::new(-dir.y, dir.x).normalize();
        }
        if normal.hypot() < 1e-6 {
            normal = Vec2::new(0.0, 1.0);
        }
        normals.push(normal);
    }

    // For each segment, create two triangles.
    // We'll also interpolate pressure and wetness for each vertex.
    for i in 0..(samples.len() - 1) {
        let p0 = samples[i];
        let p1 = samples[i + 1];
        let n0 = normals[i];
        let n1 = normals[i + 1];

        // Width at each point (pressure‑modulated).
        let w0 = half_width * (0.5 + p0.pressure * 0.5);
        let w1 = half_width * (0.5 + p1.pressure * 0.5);

        // Compute four vertices: left0, left1, right0, right1.
        let l0 = p0.position + n0 * w0;
        let r0 = p0.position - n0 * w0;
        let l1 = p1.position + n1 * w1;
        let r1 = p1.position - n1 * w1;

        // UVs: we'll compute a rough u as distance along the stroke, v as across.
        // For now, set u = i / len, v = 0/1 for left/right.
        let u0 = i as f32 / (samples.len() - 1) as f32;
        let u1 = (i + 1) as f32 / (samples.len() - 1) as f32;

        let pressure0 = p0.pressure;
        let pressure1 = p1.pressure;
        let wetness0 = wetness * (0.8 + 0.2 * p0.pressure);
        let wetness1 = wetness * (0.8 + 0.2 * p1.pressure);

        // Vertex positions are 3D (z=0).
        let to_vertex = |pos: Point, u: f32, v: f32, pressure: f32, wet: f32| InkVertex {
            position: [pos.x as f32, pos.y as f32, 0.0],
            uv: [u * paper_texture_scale, v * paper_texture_scale],
            pressure,
            wetness: wet,
            normal: [0.0, 0.0], // fill later
        };

        // Build vertices for the quad.
        let base_idx = verts.len() as u32;

        // Left side (v = 0), right side (v = 1)
        verts.push(to_vertex(l0, u0, 0.0, pressure0, wetness0));
        verts.push(to_vertex(r0, u0, 1.0, pressure0, wetness0));
        verts.push(to_vertex(l1, u1, 0.0, pressure1, wetness1));
        verts.push(to_vertex(r1, u1, 1.0, pressure1, wetness1));

        // Two triangles: (0,1,2) and (1,3,2)
        idxs.push(base_idx);
        idxs.push(base_idx + 1);
        idxs.push(base_idx + 2);
        idxs.push(base_idx + 1);
        idxs.push(base_idx + 3);
        idxs.push(base_idx + 2);
    }

    // Compute normals (optional, for lighting).
    for v in verts.iter_mut() {
        // For a flat 2D mesh, normals point in Z.
        v.normal = [0.0, 0.0];
    }

    InkMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Helper to build a kurbo path from samples with Catmull‑Rom interpolation.
fn build_kurbo_path(samples: &[PointSample]) -> BezPath {
    let mut path = BezPath::new();
    if samples.is_empty() {
        return path;
    }
    // Start at the first point.
    let pts: Vec<Point> = samples.iter().map(|s| s.position).collect();

    // Use Catmull‑Rom interpolation for smooth curves.
    // For simplicity, we'll just use line segments if we have few points.
    if pts.len() < 4 {
        for p in pts {
            path.line_to(p);
        }
        return path;
    }

    // Build a smooth curve using cubic bezier segments.
    // We'll use the standard Catmull‑Rom to Bezier conversion.
    // This is a simplified version; we can also just use line segments
    // because our samples are dense (2px step).
    // For now, just connect with lines (which is fine for dense samples).
    path.move_to(pts[0]);
    for p in &pts[1..] {
        path.line_to(*p);
    }
    path
}

/// Convert a kurbo BezPath to a lyon Path.
fn convert_kurbo_to_lyon(kurbo_path: &BezPath) -> LyonPath {
    let mut builder = LyonPath::builder();
    for el in kurbo_path.elements() {
        match el {
            PathEl::MoveTo(p) => {
                builder.move_to(LyonPoint::new(p.x as f32, p.y as f32));
            }
            PathEl::LineTo(p) => {
                builder.line_to(LyonPoint::new(p.x as f32, p.y as f32));
            }
            PathEl::QuadTo(p1, p2) => {
                builder.quadratic_bezier_to(
                    LyonPoint::new(p1.x as f32, p1.y as f32),
                    LyonPoint::new(p2.x as f32, p2.y as f32),
                );
            }
            PathEl::CurveTo(p1, p2, p3) => {
                builder.cubic_bezier_to(
                    LyonPoint::new(p1.x as f32, p1.y as f32),
                    LyonPoint::new(p2.x as f32, p2.y as f32),
                    LyonPoint::new(p3.x as f32, p3.y as f32),
                );
            }
            PathEl::ClosePath => {
                builder.close();
            }
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hw_physics::PointSample;

    #[test]
    fn test_mesh_build() {
        let mut samples = Vec::new();
        for i in 0..20 {
            let x = i as f64 * 5.0;
            let y = (i as f64 * 0.5).sin() * 10.0;
            samples.push(PointSample {
                position: Point::new(x, y),
                pressure: 0.5 + 0.5 * (i as f64 / 20.0),
                tilt_x: 0.0,
                tilt_y: 0.0,
                timestamp: 0.0,
            });
        }
        let mesh = build_ink_mesh(&samples, 2.0, 0.7, 1.0);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        // Each segment produces 4 vertices and 6 indices.
        assert_eq!(mesh.vertices.len(), (samples.len() - 1) * 4);
        assert_eq!(mesh.indices.len(), (samples.len() - 1) * 6);
    }
}
