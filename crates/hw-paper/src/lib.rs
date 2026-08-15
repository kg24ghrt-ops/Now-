//! Procedural paper texture generation using simplex noise.
//! Produces a grayscale texture with fiber‑like grain, plus a normal map.

// FIX: removed unused `Seedable` import
use noise::{NoiseFn, Perlin};
use serde::{Deserialize, Serialize};

/// Parameters for paper texture generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperParams {
    pub width: u32,
    pub height: u32,
    pub seed: u32,
    pub fiber_scale: f64, // frequency of fiber noise
    pub grain_scale: f64, // fine grain frequency
    pub fiber_strength: f64, // 0..1
    pub grain_strength: f64, // 0..1
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
        }
    }
}

/// A generated paper texture, returned as raw grayscale data (u8, 0..255).
pub struct PaperTexture {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>, // grayscale, row‑major
    pub normal_map: Vec<f32>, // (nx, ny, nz) as f32, row‑major
}

impl PaperTexture {
    /// Generate a new paper texture from parameters.
    pub fn generate(params: &PaperParams) -> Self {
        let width = params.width;
        let height = params.height;
        let mut data = vec![0u8; (width * height) as usize];
        let mut normal_map = vec![0.0f32; (width * height * 3) as usize];

        let fiber_noise = Perlin::new(params.seed);
        let grain_noise = Perlin::new(params.seed.wrapping_add(100));
        let normal_noise = Perlin::new(params.seed.wrapping_add(200));

        // Pre‑compute noise for each pixel.
        for y in 0..height {
            for x in 0..width {
                let px = x as f64 / width as f64;
                let py = y as f64 / height as f64;

                // Fiber noise: anisotropic, elongated in one direction.
                let fiber = fiber_noise.get([px * params.fiber_scale, py * params.fiber_scale, 0.0]);
                let fiber_val = (fiber * 0.5 + 0.5) * params.fiber_strength;

                // Grain noise: isotropic high‑frequency.
                let grain = grain_noise.get([px * params.grain_scale, py * params.grain_scale, 0.0]);
                let grain_val = (grain * 0.5 + 0.5) * params.grain_strength;

                // Combine.
                let mut val = fiber_val + grain_val;
                val = val.clamp(0.0, 1.0);

                // Store as u8.
                data[(y * width + x) as usize] = (val * 255.0) as u8;

                // Normal map: compute gradient of the noise.
                let eps = 1.0 / width as f64;
                let dx = normal_noise.get([(px + eps) * params.fiber_scale, py * params.fiber_scale, 0.0])
                    - normal_noise.get([(px - eps) * params.fiber_scale, py * params.fiber_scale, 0.0]);
                let dy = normal_noise.get([px * params.fiber_scale, (py + eps) * params.fiber_scale, 0.0])
                    - normal_noise.get([px * params.fiber_scale, (py - eps) * params.fiber_scale, 0.0]);

                let nx = -(dx * 0.2) as f32;
                let ny = -(dy * 0.2) as f32;
                let nz = 1.0;
                let len = (nx * nx + ny * ny + nz * nz).sqrt();
                let idx = (y * width + x) as usize * 3;
                normal_map[idx] = nx / len;
                normal_map[idx + 1] = ny / len;
                normal_map[idx + 2] = nz / len;
            }
        }

        Self {
            width,
            height,
            data,
            normal_map,
        }
    }

    /// Create an RGBA version (for GPU upload).
    pub fn as_rgba(&self) -> Vec<u8> {
        let mut rgba = vec![0u8; (self.width * self.height * 4) as usize];
        for i in 0..self.data.len() {
            let idx = i * 4;
            let v = self.data[i];
            rgba[idx] = v;
            rgba[idx + 1] = v;
            rgba[idx + 2] = v;
            rgba[idx + 3] = 255;
        }
        rgba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate() {
        let params = PaperParams::default();
        let tex = PaperTexture::generate(&params);
        assert_eq!(tex.data.len(), (1024 * 1024) as usize);
        assert_eq!(tex.normal_map.len(), (1024 * 1024 * 3) as usize);
    }
}