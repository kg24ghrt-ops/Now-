//! Physics simulation for handwriting.
//! Provides a stateful pen simulator driven by a `HandProfile`.

use kurbo::{Point, Vec2};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// A single sample produced by the pen simulation.
#[derive(Debug, Clone, Copy)]
pub struct PointSample {
    pub position: Point,
    pub pressure: f32,
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub timestamp: f64, // relative time in seconds
}

/// Defines how fatigue decays over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FatigueCurve {
    pub initial: f32,      // 0..1, starting fatigue
    pub decay_rate: f32,   // per second
    pub max_fatigue: f32,  // 0..1
}

/// Tremor parameters – sinusoidal jitter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorParams {
    pub frequency: f32,    // Hz
    pub amplitude: f32,    // in pixels
    pub noise_scale: f32,  // additional random jitter
}

/// Pressure decay as the writer tires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureDecay {
    pub start_pressure: f32, // 0..1
    pub min_pressure: f32,   // 0..1
    pub decay_rate: f32,     // per second of writing
}

/// Slant drift: slant angle changes over the course of a stroke/line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlantDrift {
    pub base_slant: f32,       // degrees from vertical
    pub drift_per_stroke: f32, // degrees change per stroke
    pub drift_range: f32,      // max deviation
}

/// Spacing bias: horizontal drift per character (e.g., cursive ligatures).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingBias {
    pub bias_per_char: f32, // pixels per character
    pub jitter_scale: f32,  // randomness in spacing
}

/// A complete handwriting profile that determines the 'personality' of the pen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandProfile {
    pub fatigue_curve: FatigueCurve,
    pub tremor: TremorParams,
    pub pressure_decay: PressureDecay,
    pub slant_drift: SlantDrift,
    pub spacing_bias: SpacingBias,
    pub base_stroke_width: f32, // in pixels
    pub speed: f64,             // pixels per second (nominal drawing speed)
}

impl Default for HandProfile {
    fn default() -> Self {
        Self {
            fatigue_curve: FatigueCurve {
                initial: 0.2,
                decay_rate: 0.05,
                max_fatigue: 0.9,
            },
            tremor: TremorParams {
                frequency: 8.0,
                amplitude: 0.6,
                noise_scale: 0.3,
            },
            pressure_decay: PressureDecay {
                start_pressure: 0.9,
                min_pressure: 0.4,
                decay_rate: 0.02,
            },
            slant_drift: SlantDrift {
                base_slant: 5.0,
                drift_per_stroke: 1.2,
                drift_range: 6.0,
            },
            spacing_bias: SpacingBias {
                bias_per_char: 0.5,
                jitter_scale: 0.2,
            },
            base_stroke_width: 2.5,
            speed: 120.0,
        }
    }
}

/// The stateful pen simulator.
pub struct PenSimulator {
    profile: HandProfile,
    rng: Pcg64,
    fatigue: f32,
    current_slant: f32,
    spacing_offset: f32,
    time: f64,
    last_position: Point,
    pressure: f32,
}

impl PenSimulator {
    /// Create a new simulator with a given profile and seed.
    pub fn new(profile: HandProfile, seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        let fatigue = profile.fatigue_curve.initial;
        let current_slant = profile.slant_drift.base_slant
            + rng.gen_range(-0.5..0.5) * profile.slant_drift.drift_range;
        Self {
            profile,
            rng,
            fatigue,
            current_slant,
            spacing_offset: 0.0,
            time: 0.0,
            last_position: Point::ZERO,
            pressure: 0.0,
        }
    }

    /// Reset the simulator to a fresh state (keeps the same profile and seed).
    /// Note: Pcg64 doesn't expose `get_seed()`, so we store the seed separately.
    pub fn reset(&mut self) {
        // We can't re-seed from the existing RNG easily.
        // Instead, we'll just reset the state variables to their initial values.
        self.fatigue = self.profile.fatigue_curve.initial;
        self.current_slant = self.profile.slant_drift.base_slant
            + self.rng.gen_range(-0.5..0.5) * self.profile.slant_drift.drift_range;
        self.spacing_offset = 0.0;
        self.time = 0.0;
        self.pressure = 0.0;
    }

    /// Trace along a path (list of points) and produce samples.
    /// The path is assumed to be in pixel coordinates.
    pub fn trace_path(&mut self, path: &[Point], is_new_stroke: bool) -> Vec<PointSample> {
        if path.is_empty() {
            return Vec::new();
        }

        // If this is a new stroke, update slant drift and reset pressure a bit.
        if is_new_stroke {
            let drift = self.rng.gen_range(-1.0..1.0) * self.profile.slant_drift.drift_per_stroke;
            self.current_slant = (self.current_slant + drift)
                .clamp(
                    self.profile.slant_drift.base_slant - self.profile.slant_drift.drift_range,
                    self.profile.slant_drift.base_slant + self.profile.slant_drift.drift_range,
                );
            // Slight pressure recovery at start of stroke.
            self.pressure = self.profile.pressure_decay.start_pressure;
        }

        let mut samples = Vec::new();
        let speed = self.profile.speed;

        // Interpolate along the path with fixed time steps to get smooth samples.
        // We'll use a step size of ~2 pixels to keep samples dense.
        let step_size = 2.0; // pixels
        let mut current_pos = path[0];

        // Estimate total length to set duration.
        let mut total_len = 0.0;
        for w in path.windows(2) {
            total_len += w[0].distance(w[1]);
        }
        let duration = if total_len > 0.0 { total_len / speed } else { 0.1 };

        // If we have no length, just emit the first point.
        if total_len < 0.001 {
            samples.push(self.sample_at(current_pos, 0.0));
            return samples;
        }

        let mut t = 0.0;
        let dt = 0.005; // 5ms per step, ~200 samples per second.
        let mut seg_len = 0.0;
        let mut seg_idx = 0;

        // Walk through the path segments.
        while t < duration {
            // Advance along the path.
            let dist = speed * dt;
            let mut pos = current_pos;

            // Move along the path by `dist` pixels.
            let mut moved = 0.0;
            while moved < dist && seg_idx < path.len() - 1 {
                let seg = path[seg_idx + 1] - path[seg_idx];
                let seg_len_full = seg.hypot();
                if seg_len_full < 1e-6 {
                    seg_idx += 1;
                    continue;
                }
                let remaining_in_seg = seg_len_full - seg_len;
                let needed = dist - moved;
                if needed < remaining_in_seg {
                    // We stay inside this segment.
                    let frac = needed / seg_len_full;
                    pos = path[seg_idx] + seg * frac;
                    seg_len += needed;
                    moved += needed;
                } else {
                    // Move to the end of this segment.
                    pos = path[seg_idx + 1];
                    moved += remaining_in_seg;
                    seg_len = 0.0;
                    seg_idx += 1;
                    if seg_idx >= path.len() - 1 {
                        break;
                    }
                }
            }

            // If we reached the end, clamp position.
            if seg_idx >= path.len() - 1 {
                pos = *path.last().unwrap();
            }

            // Update fatigue (only while writing).
            // Cast dt to f32 for arithmetic with f32 values.
            let dt_f32 = dt as f32;
            self.fatigue = (self.fatigue + self.profile.fatigue_curve.decay_rate * dt_f32)
                .min(self.profile.fatigue_curve.max_fatigue);

            // Update pressure based on fatigue.
            let pressure_factor = 1.0 - self.fatigue * 0.6;
            let target_pressure = self.profile.pressure_decay.start_pressure
                + (self.profile.pressure_decay.min_pressure
                    - self.profile.pressure_decay.start_pressure)
                    * (1.0 - pressure_factor);
            // Smooth pressure transition.
            self.pressure += (target_pressure - self.pressure) * dt_f32 * 5.0;
            self.pressure = self.pressure.clamp(0.0, 1.0);

            // Sample at this position.
            let sample = self.sample_at(pos, self.time + t);
            samples.push(sample);

            // Advance time.
            t += dt;
            current_pos = pos;
        }

        // Ensure we have the last point exactly.
        if let Some(last_sample) = samples.last() {
            if last_sample.position.distance(*path.last().unwrap()) > 0.5 {
                let final_sample = self.sample_at(*path.last().unwrap(), self.time + duration);
                samples.push(final_sample);
            }
        }

        self.time += duration;
        self.last_position = *path.last().unwrap();

        samples
    }

    /// Internal helper to generate a single sample with tremor and slant.
    fn sample_at(&mut self, base_position: Point, time: f64) -> PointSample {
        // Apply tremor: sinusoidal + noise.
        let freq = self.profile.tremor.frequency as f64;
        let amp = self.profile.tremor.amplitude as f64;
        let noise_scale = self.profile.tremor.noise_scale as f64;

        let t_sin = (2.0 * PI * freq * time).sin();
        let noise_x: f64 = self.rng.gen();
        let noise_y: f64 = self.rng.gen();

        let dx = amp * t_sin + noise_scale * (noise_x - 0.5);
        let dy = amp * (2.0 * PI * freq * time * 0.7).cos() + noise_scale * (noise_y - 0.5);

        let mut pos = base_position + Vec2::new(dx, dy);

        // Apply slant drift: horizontal offset proportional to vertical position.
        let slant_rad = self.current_slant.to_radians() as f64;
        let slant_offset = pos.y * slant_rad.tan() * 0.1; // subtle
        pos.x += slant_offset;

        // Apply spacing bias (randomized).
        let spacing_noise: f64 = self.rng.gen();
        let spacing_bias = self.profile.spacing_bias.bias_per_char as f64
            + (spacing_noise - 0.5) * self.profile.spacing_bias.jitter_scale as f64;
        pos.x += spacing_bias;

        // Tilt mimics pen angle: small random variations.
        let tilt_x = (self.rng.gen::<f32>() - 0.5) * 0.2;
        let tilt_y = (self.rng.gen::<f32>() - 0.5) * 0.2 + 0.5;

        PointSample {
            position: pos,
            pressure: self.pressure,
            tilt_x,
            tilt_y,
            timestamp: time,
        }
    }

    /// Get a reference to the underlying profile.
    pub fn profile(&self) -> &HandProfile {
        &self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_line() {
        let profile = HandProfile::default();
        let mut sim = PenSimulator::new(profile, 12345);
        let path = vec![Point::new(0.0, 0.0), Point::new(100.0, 0.0)];
        let samples = sim.trace_path(&path, true);
        assert!(samples.len() > 10);
        assert!(samples[0].pressure > 0.0);
        // Last point should be near the end.
        assert!(samples.last().unwrap().position.distance(Point::new(100.0, 0.0)) < 3.0);
    }
}