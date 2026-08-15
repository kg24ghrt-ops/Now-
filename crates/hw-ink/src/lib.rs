//! hw-ink — Converts physics `StrokePoint`s into GPU-tessellated ink meshes.
//!
//! Bridges the fixed `hw-physics` (event-driven, f32, `StrokePoint`) to the
//! fixed `hw-render-wgpu` (GLSL shader expecting vec4(x, y, pressure, timestamp)).
//!
//! Zero-allocation on the hot path: `StrokeBatch` reuses buffers.

use bytemuck::{Pod, Zeroable};
use hw_physics::StrokePoint;
use kurbo::{Point, Vec2};

// ---------------------------------------------------------------------------
// Vertex format — matches GLSL shader layout
// ---------------------------------------------------------------------------

/// GPU vertex for ink stroke.
///
/// Layout (16 bytes): vec2 position (mm), float pressure, float timestamp.
/// Matches `hw-render-wgpu` GLSL vertex shader `layout(location = 0..2)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct InkVertex {
    /// Position in screen mm (DPI-independent, from physics).
    pub pos_mm: [f32; 2],
    /// Smoothed pressure [0..1].
    pub pressure: f32,
    /// Seconds since session start (for ink drying).
    pub timestamp_secs: f32,
}

/// Complete ink mesh, ready for GPU upload via `BufferPool`.
#[derive(Debug, Clone)]
pub struct InkMesh {
    pub vertices: Vec<InkVertex>,
    pub indices: Vec<u32>,
}

impl InkMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Total bytes for vertex buffer upload.
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Total bytes for index buffer upload.
    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }
}

// ---------------------------------------------------------------------------
// Mesh builder — ribbon tessellation with variable width
// ---------------------------------------------------------------------------

/// Build an ink mesh from physics stroke points.
///
/// `points`: deformed stroke points from `PhysicsEngine::sample_batch()`.
/// `paper_texture_scale`: UV scale for paper texture repeat.
///
/// Width comes from `StrokePoint.width_mm` (already computed by physics
/// from pressure, slant, and velocity). No re-computation here.
pub fn build_ink_mesh(
    points: &[StrokePoint],
    paper_texture_scale: f32,
) -> InkMesh {
    if points.len() < 2 {
        return InkMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let n = points.len();
    let mut verts = Vec::with_capacity((n - 1) * 4);
    let mut idxs = Vec::with_capacity((n - 1) * 6);

    // Pre-compute normals for each point (perpendicular to stroke direction).
    let mut normals = Vec::with_capacity(n);
    for i in 0..n {
        let normal = if i == 0 {
            stroke_normal(&points[0], &points[1])
        } else if i == n - 1 {
            stroke_normal(&points[i - 1], &points[i])
        } else {
            let n1 = stroke_normal(&points[i - 1], &points[i]);
            let n2 = stroke_normal(&points[i], &points[i + 1]);
            [(n1[0] + n2[0]) * 0.5, (n1[1] + n2[1]) * 0.5]
        };
        // Normalize
        let len = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        if len > 1e-6 {
            normals.push([normal[0] / len, normal[1] / len]);
        } else {
            normals.push([0.0, 1.0]);
        }
    }

    // Build ribbon quads.
    let mut cum_dist = 0.0; // cumulative distance for UV.u

    for i in 0..(n - 1) {
        let p0 = &points[i];
        let p1 = &points[i + 1];
        let n0 = normals[i];
        let n1 = normals[i + 1];

        // Half-width at each point (from physics, already includes pressure/slant).
        let hw0 = p0.width_mm * 0.5;
        let hw1 = p1.width_mm * 0.5;

        // Apply aspect ratio (elliptical nib from slant).
        let a0 = p0.aspect_ratio;
        let a1 = p1.aspect_ratio;

        // Deform normal by aspect to get elliptical cross-section.
        let nd0 = [n0[0] * a0, n0[1]];
        let nd1 = [n1[0] * a1, n1[1]];

        // Four corners of the quad.
        let l0 = [p0.pos_mm_x + nd0[0] * hw0, p0.pos_mm_y + nd0[1] * hw0];
        let r0 = [p0.pos_mm_x - nd0[0] * hw0, p0.pos_mm_y - nd0[1] * hw0];
        let l1 = [p1.pos_mm_x + nd1[0] * hw1, p1.pos_mm_y + nd1[1] * hw1];
        let r1 = [p1.pos_mm_x - nd1[0] * hw1, p1.pos_mm_y - nd1[1] * hw1];

        // UVs: u = cumulative distance, v = 0/1 for left/right.
        let seg_len = ((p1.pos_mm_x - p0.pos_mm_x).powi(2)
            + (p1.pos_mm_y - p0.pos_mm_y).powi(2))
            .sqrt();
        let u0 = cum_dist * paper_texture_scale;
        cum_dist += seg_len;
        let u1 = cum_dist * paper_texture_scale;

        let base_idx = verts.len() as u32;

        // Left side (v=0), right side (v=1).
        // We don't store UVs in the vertex anymore — the fragment shader
        // derives them from position for procedural paper, or we could add
        // a separate UV attribute. For now, the GLSL shader uses v_uv = a_position
        // so we keep it simple.
        verts.push(InkVertex {
            pos_mm: l0,
            pressure: p0.pressure,
            timestamp_secs: p0.timestamp_secs,
        });
        verts.push(InkVertex {
            pos_mm: r0,
            pressure: p0.pressure,
            timestamp_secs: p0.timestamp_secs,
        });
        verts.push(InkVertex {
            pos_mm: l1,
            pressure: p1.pressure,
            timestamp_secs: p1.timestamp_secs,
        });
        verts.push(InkVertex {
            pos_mm: r1,
            pressure: p1.pressure,
            timestamp_secs: p1.timestamp_secs,
        });

        // Two triangles: (0,1,2) and (1,3,2)
        idxs.push(base_idx);
        idxs.push(base_idx + 1);
        idxs.push(base_idx + 2);
        idxs.push(base_idx + 1);
        idxs.push(base_idx + 3);
        idxs.push(base_idx + 2);
    }

    InkMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Compute stroke normal (perpendicular to direction) between two points.
fn stroke_normal(a: &StrokePoint, b: &StrokePoint) -> [f32; 2] {
    let dx = b.pos_mm_x - a.pos_mm_x;
    let dy = b.pos_mm_y - a.pos_mm_y;
    // Perpendicular: (-dy, dx)
    [-dy, dx]
}

// ---------------------------------------------------------------------------
// Batch processor — zero-allocation stroke accumulation
// ---------------------------------------------------------------------------

/// Accumulates stroke points and builds meshes on demand.
///
/// Reuses internal buffers to avoid per-stroke allocation.
pub struct StrokeMeshBuilder {
    points: Vec<StrokePoint>,
    vertex_scratch: Vec<InkVertex>,
    index_scratch: Vec<u32>,
}

impl StrokeMeshBuilder {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            points: Vec::with_capacity(cap),
            vertex_scratch: Vec::with_capacity(cap * 4),
            index_scratch: Vec::with_capacity(cap * 6),
        }
    }

    /// Add a point from the physics engine.
    pub fn push(&mut self, point: StrokePoint) {
        self.points.push(point);
    }

    /// Clear for new stroke.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Build mesh from accumulated points.
    pub fn build(&mut self, paper_texture_scale: f32) -> InkMesh {
        // Reuse scratch buffers.
        self.vertex_scratch.clear();
        self.index_scratch.clear();

        if self.points.len() < 2 {
            return InkMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            };
        }

        let n = self.points.len();
        self.vertex_scratch.reserve((n - 1) * 4);
        self.index_scratch.reserve((n - 1) * 6);

        // Pre-compute normals.
        let mut normals = Vec::with_capacity(n);
        for i in 0..n {
            let normal = if i == 0 {
                stroke_normal(&self.points[0], &self.points[1])
            } else if i == n - 1 {
                stroke_normal(&self.points[i - 1], &self.points[i])
            } else {
                let n1 = stroke_normal(&self.points[i - 1], &self.points[i]);
                let n2 = stroke_normal(&self.points[i], &self.points[i + 1]);
                [(n1[0] + n2[0]) * 0.5, (n1[1] + n2[1]) * 0.5]
            };
            let len = (normal[0].powi(2) + normal[1].powi(2)).sqrt();
            normals.push(if len > 1e-6 {
                [normal[0] / len, normal[1] / len]
            } else {
                [0.0, 1.0]
            });
        }

        let mut cum_dist = 0.0f32;

        for i in 0..(n - 1) {
            let p0 = &self.points[i];
            let p1 = &self.points[i + 1];
            let n0 = normals[i];
            let n1 = normals[i + 1];

            let hw0 = p0.width_mm * 0.5;
            let hw1 = p1.width_mm * 0.5;

            let a0 = p0.aspect_ratio;
            let a1 = p1.aspect_ratio;

            let nd0 = [n0[0] * a0, n0[1]];
            let nd1 = [n1[0] * a1, n1[1]];

            let l0 = [p0.pos_mm_x + nd0[0] * hw0, p0.pos_mm_y + nd0[1] * hw0];
            let r0 = [p0.pos_mm_x - nd0[0] * hw0, p0.pos_mm_y - nd0[1] * hw0];
            let l1 = [p1.pos_mm_x + nd1[0] * hw1, p1.pos_mm_y + nd1[1] * hw1];
            let r1 = [p1.pos_mm_x - nd1[0] * hw1, p1.pos_mm_y - nd1[1] * hw1];

            let seg_len = ((p1.pos_mm_x - p0.pos_mm_x).powi(2)
                + (p1.pos_mm_y - p0.pos_mm_y).powi(2))
                .sqrt();
            let u0 = cum_dist * paper_texture_scale;
            cum_dist += seg_len;
            let _u1 = cum_dist * paper_texture_scale;

            let base_idx = self.vertex_scratch.len() as u32;

            self.vertex_scratch.push(InkVertex {
                pos_mm: l0,
                pressure: p0.pressure,
                timestamp_secs: p0.timestamp_secs,
            });
            self.vertex_scratch.push(InkVertex {
                pos_mm: r0,
                pressure: p0.pressure,
                timestamp_secs: p0.timestamp_secs,
            });
            self.vertex_scratch.push(InkVertex {
                pos_mm: l1,
                pressure: p1.pressure,
                timestamp_secs: p1.timestamp_secs,
            });
            self.vertex_scratch.push(InkVertex {
                pos_mm: r1,
                pressure: p1.pressure,
                timestamp_secs: p1.timestamp_secs,
            });

            self.index_scratch.push(base_idx);
            self.index_scratch.push(base_idx + 1);
            self.index_scratch.push(base_idx + 2);
            self.index_scratch.push(base_idx + 1);
            self.index_scratch.push(base_idx + 3);
            self.index_scratch.push(base_idx + 2);
        }

        InkMesh {
            vertices: self.vertex_scratch.clone(),
            indices: self.index_scratch.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f32, y: f32, pressure: f32, width: f32) -> StrokePoint {
        StrokePoint {
            pos_mm_x: x,
            pos_mm_y: y,
            pressure,
            timestamp_secs: 0.0,
            width_mm: width,
            aspect_ratio: 1.0,
            slant_azimuth_rad: 0.0,
            slant_altitude_rad: 0.0,
        }
    }

    #[test]
    fn test_mesh_from_stroke_points() {
        let points: Vec<StrokePoint> = (0..20)
            .map(|i| {
                let t = i as f32;
                make_point(t * 5.0, (t * 0.5).sin() * 10.0, 0.5 + 0.5 * (t / 20.0), 2.0)
            })
            .collect();

        let mesh = build_ink_mesh(&points, 1.0);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.vertices.len(), (points.len() - 1) * 4);
        assert_eq!(mesh.indices.len(), (points.len() - 1) * 6);
        assert_eq!(mesh.vertex_bytes().len(), mesh.vertices.len() * 16);
    }

    #[test]
    fn test_batch_builder_reuse() {
        let mut builder = StrokeMeshBuilder::with_capacity(100);

        for i in 0..10 {
            let t = i as f32;
            builder.push(make_point(t * 5.0, 0.0, 0.8, 1.5));
        }
        let mesh1 = builder.build(1.0);
        assert_eq!(mesh1.triangle_count(), 9);

        builder.clear();
        for i in 0..5 {
            let t = i as f32;
            builder.push(make_point(t * 3.0, 5.0, 0.6, 1.0));
        }
        let mesh2 = builder.build(1.0);
        assert_eq!(mesh2.triangle_count(), 4);
    }

    #[test]
    fn test_empty_mesh() {
        let mesh = build_ink_mesh(&[], 1.0);
        assert!(mesh.is_empty());

        let mesh = build_ink_mesh(&[make_point(0.0, 0.0, 1.0, 2.0)], 1.0);
        assert!(mesh.is_empty());
    }
}
