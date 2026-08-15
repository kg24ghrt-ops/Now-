//! hw-paper — Procedural paper texture generation with GPU-ready output.
//!
//! Produces:
//! - Diffuse/albedo texture (RGBA8, sRGB) — paper color with fiber grain.
//! - Normal map (RGBA8, linear) — surface normals for ink lighting.
//!
//! Both outputs are in the exact format `hw-render-wgpu` expects for
//! `upload_paper_textures()`.

use noise::{NoiseFn, Perlin};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg32;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperParams {
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    /// Frequency of large fiber patterns.
    pub fiber_scale: f64,
    /// Frequency of fine grain.
    pub grain_scale: f64,
    /// Strength of fiber pattern [0..1].
    pub fiber_strength: f64,
    /// Strength of fine grain [0..1].
    pub grain_strength: f64,
    /// Anisotropy: how much fibers align to a direction (0 = random, 1 = aligned).
    pub fiber_anisotropy: f64,
    /// Base paper color (RGB, 0..1). Default warm white.
    pub base_color: [f32; 3],
}

impl Default for PaperParams {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            seed: 42,
            fiber_scale: 4.0,
            grain_scale: 20.0,
            fiber_strength: 0.6,
            grain_strength: 0.3,
            fiber_anisotropy: 0.7,
            base_color: [0.98, 0.96, 0.94], // warm white
        }
    }
}

// ---------------------------------------------------------------------------
// Generated texture
// ---------------------------------------------------------------------------

pub struct PaperTexture {
    pub width: u32,
    pub height: u32,
    /// Diffuse/albedo: RGBA8, sRGB, row-major. Size = width * height * 4.
    pub diffuse_rgba: Vec<u8>,
    /// Normal map: RGBA8, linear, row-major. Size = width * height * 4.
    /// Encoded as (nx, ny, nz, 1.0) where each component is in [0, 255].
    pub normal_rgba: Vec<u8>,
}

impl PaperTexture {
    /// Generate paper texture from parameters.
    pub fn generate(params: &PaperParams) -> Self {
        let w = params.width as usize;
        let h = params.height as usize;
        let pixel_count = w * h;

        let mut diffuse = vec![0u8; pixel_count * 4];
        let mut normal = vec![0u8; pixel_count * 4];

        // Seeded noise generators.
        let fiber_noise = Perlin::new(params.seed);
        let grain_noise = Perlin::new(params.seed.wrapping_add(100));
        let normal_noise = Perlin::new(params.seed.wrapping_add(200));

        // Fiber direction (slightly random, seeded).
        let mut rng = Pcg32::seed_from_u64(params.seed as u64);
        let fiber_angle: f64 = rng.gen_range(0.0..std::f64::consts::PI);
        let fiber_dir = [fiber_angle.cos(), fiber_angle.sin()];

        let eps = 1.0 / params.width as f64;

        for y in 0..h {
            for x in 0..w {
                let px = x as f64 / w as f64;
                let py = y as f64 / h as f64;
                let idx = (y * w + x) * 4;

                // --- Diffuse ---
                // Anisotropic fiber: elongated along fiber_dir.
                let fiber_u = px * fiber_dir[0] + py * fiber_dir[1];
                let fiber_v = -px * fiber_dir[1] + py * fiber_dir[0];
                let fiber = fiber_noise.get([
                    fiber_u * params.fiber_scale,
                    fiber_v * params.fiber_scale * (1.0 + params.fiber_anisotropy * 3.0),
                    0.0,
                ]);
                let fiber_val = (fiber * 0.5 + 0.5) * params.fiber_strength;

                // Isotropic grain.
                let grain = grain_noise.get([
                    px * params.grain_scale,
                    py * params.grain_scale,
                    0.0,
                ]);
                let grain_val = (grain * 0.5 + 0.5) * params.grain_strength;

                let mut intensity = (fiber_val + grain_val).clamp(0.0, 1.0);
                // Invert: high intensity = dark fiber, low = light paper.
                intensity = 1.0 - intensity * 0.15; // subtle

                let r = (params.base_color[0] * intensity as f32 * 255.0) as u8;
                let g = (params.base_color[1] * intensity as f32 * 255.0) as u8;
                let b = (params.base_color[2] * intensity as f32 * 255.0) as u8;

                diffuse[idx] = r;
                diffuse[idx + 1] = g;
                diffuse[idx + 2] = b;
                diffuse[idx + 3] = 255;

                // --- Normal map ---
                // Compute gradient of normal_noise for surface perturbation.
                let nx0 = normal_noise.get([
                    (px - eps) * params.fiber_scale,
                    py * params.fiber_scale,
                    0.0,
                ]);
                let nx1 = normal_noise.get([
                    (px + eps) * params.fiber_scale,
                    py * params.fiber_scale,
                    0.0,
                ]);
                let ny0 = normal_noise.get([
                    px * params.fiber_scale,
                    (py - eps) * params.fiber_scale,
                    0.0,
                ]);
                let ny1 = normal_noise.get([
                    px * params.fiber_scale,
                    (py + eps) * params.fiber_scale,
                    0.0,
                ]);

                let dx = (nx1 - nx0) * 0.5;
                let dy = (ny1 - ny0) * 0.5;

                // Convert gradient to normal: (-dx, -dy, 1.0) then normalize.
                let mut nx = -(dx * 2.0) as f32;
                let mut ny = -(dy * 2.0) as f32;
                let nz = 1.0f32;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                nx /= len;
                ny /= len;
                let nz = nz / len;

                // Encode to [0, 255].
                normal[idx] = ((nx * 0.5 + 0.5) * 255.0) as u8;
                normal[idx + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
                normal[idx + 2] = ((nz * 0.5 + 0.5) * 255.0) as u8;
                normal[idx + 3] = 255;
            }
        }

        Self {
            width: params.width,
            height: params.height,
            diffuse_rgba: diffuse,
            normal_rgba: normal,
        }
    }

    /// Validate that buffers have correct size.
    pub fn validate(&self) -> Result<(), &'static str> {
        let expected = (self.width * self.height * 4) as usize;
        if self.diffuse_rgba.len() != expected {
            return Err("diffuse_rgba size mismatch");
        }
        if self.normal_rgba.len() != expected {
            return Err("normal_rgba size mismatch");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl PaperTexture {
    /// Quick generation with default params.
    pub fn default_1024() -> Self {
        Self::generate(&PaperParams::default())
    }

    /// High-res for tablets.
    pub fn high_res() -> Self {
        Self::generate(&PaperParams {
            width: 2048,
            height: 2048,
            ..Default::default()
        })
    }

    /// Rough handmade paper.
    pub fn rough() -> Self {
        Self::generate(&PaperParams {
            fiber_scale: 2.0,
            grain_scale: 40.0,
            fiber_strength: 0.8,
            grain_strength: 0.5,
            fiber_anisotropy: 0.3,
            ..Default::default()
        })
    }

    /// Smooth printer paper.
    pub fn smooth() -> Self {
        Self::generate(&PaperParams {
            fiber_scale: 8.0,
            grain_scale: 10.0,
            fiber_strength: 0.3,
            grain_strength: 0.1,
            fiber_anisotropy: 0.9,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_default() {
        let tex = PaperTexture::generate(&PaperParams::default());
        assert_eq!(tex.diffuse_rgba.len(), 1024 * 1024 * 4);
        assert_eq!(tex.normal_rgba.len(), 1024 * 1024 * 4);
        assert!(tex.validate().is_ok());
    }

    #[test]
    fn test_presets() {
        let rough = PaperTexture::rough();
        let smooth = PaperTexture::smooth();
        assert_eq!(rough.width, 1024);
        assert_eq!(smooth.width, 1024);
    }

    #[test]
    fn test_normal_encoding() {
        let params = PaperParams {
            width: 4,
            height: 4,
            ..Default::default()
        };
        let tex = PaperTexture::generate(&params);
        // Check that normals are in valid range [0, 255].
        for i in 0..tex.normal_rgba.len() {
            assert!(tex.normal_rgba[i] <= 255);
        }
        // Center pixel should have near-neutral normal (128, 128, ~255).
        let center = (2 * 4 + 2) * 4;
        assert!(tex.normal_rgba[center] >= 120 && tex.normal_rgba[center] <= 136);
        assert!(tex.normal_rgba[center + 1] >= 120 && tex.normal_rgba[center + 1] <= 136);
        assert!(tex.normal_rgba[center + 2] >= 240);
    }
}
