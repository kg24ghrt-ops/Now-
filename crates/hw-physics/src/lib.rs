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
    pub initial: f32,     // 0..1, starting fatigue
    pub decay_rate: f32,  // per second
    pub max_fatigue: f32, // 0..1
}

/// Tremor parameters – sinusoidal jitter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorParams {
    pub frequency: f32,   // Hz
    pub amplitude: f32,   // in pixels
    pub noise_scale: f32, // additional random jitter
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
    pub fn reset(&mut self) {
        self.fatigue = self.profile.fatigue_curve.initial;
        self.current_slant = self.profile.slant_drift.base_slant
            + self.rng.gen_range(-0.5..0.5) * self.profile.slant_drift.drift_range;
        self.spacing_offset = 0.0;
        self.time = 0.0;
        self.pressure = 0.0;
    }

    /// Trace along a path (list of points) and produce samples.
    pub fn trace_path(&mut self, path: &[Point], is_new_stroke: bool) -> Vec<PointSample> {
        if path.is_empty() {
            return Vec::new();
        }

        if is_new_stroke {
            let drift = self.rng.gen_range(-1.0..1.0) * self.profile.slant_drift.drift_per_stroke;
            self.current_slant = (self.current_slant + drift).clamp(
                self.profile.slant_drift.base_slant - self.profile.slant_drift.drift_range,
                self.profile.slant_drift.base_slant + self.profile.slant_drift.drift_range,
            );
            self.pressure = self.profile.pressure_decay.start_pressure;
        }

        let mut samples = Vec::new();
        let speed = self.profile.speed;

        let mut total_len = 0.0;
        for w in path.windows(2) {
            total_len += w[0].distance(w[1]);
        }
        let duration = if total_len > 0.0 {
            total_len / speed
        } else {
            0.1
        };

        if total_len < 0.001 {
            samples.push(self.sample_at(path[0], 0.0));
            return samples;
        }

        let mut t = 0.0;
        let dt = 0.005;
        let mut seg_len = 0.0;
        let mut seg_idx = 0;
        let mut current_pos = path[0];

        while t < duration {
            let dist = speed * dt;
            let mut pos = current_pos;
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
                    let frac = needed / seg_len_full;
                    pos = path[seg_idx] + seg * frac;
                    seg_len += needed;
                    moved += needed;
                } else {
                    pos = path[seg_idx + 1];
                    moved += remaining_in_seg;
                    seg_len = 0.0;
                    seg_idx += 1;
                    if seg_idx >= path.len() - 1 {
                        break;
                    }
                }
            }

            if seg_idx >= path.len() - 1 {
                pos = *path.last().unwrap();
            }

            let dt_f32 = dt as f32;
            self.fatigue = (self.fatigue + self.profile.fatigue_curve.decay_rate * dt_f32)
                .min(self.profile.fatigue_curve.max_fatigue);

            let pressure_factor = 1.0 - self.fatigue * 0.6;
            let target_pressure = self.profile.pressure_decay.start_pressure
                + (self.profile.pressure_decay.min_pressure
                    - self.profile.pressure_decay.start_pressure)
                    * (1.0 - pressure_factor);
            self.pressure += (target_pressure - self.pressure) * dt_f32 * 5.0;
            self.pressure = self.pressure.clamp(0.0, 1.0);

            let sample = self.sample_at(pos, self.time + t);
            samples.push(sample);

            t += dt;
            current_pos = pos;
        }

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

    fn sample_at(&mut self, base_position: Point, time: f64) -> PointSample {
        let freq = self.profile.tremor.frequency as f64;
        let amp = self.profile.tremor.amplitude as f64;
        let noise_scale = self.profile.tremor.noise_scale as f64;

        let t_sin = (2.0 * PI * freq * time).sin();
        let noise_x: f64 = self.rng.gen();
        let noise_y: f64 = self.rng.gen();

        let dx = amp * t_sin + noise_scale * (noise_x - 0.5);
        let dy = amp * (2.0 * PI * freq * time * 0.7).cos() + noise_scale * (noise_y - 0.5);

        let mut pos = base_position + Vec2::new(dx, dy);

        let slant_rad = self.current_slant.to_radians() as f64;
        let slant_offset = pos.y * slant_rad.tan() * 0.1;
        pos.x += slant_offset;

        let spacing_noise: f64 = self.rng.gen();
        let spacing_bias = self.profile.spacing_bias.bias_per_char as f64
            + (spacing_noise - 0.5) * self.profile.spacing_bias.jitter_scale as f64;
        pos.x += spacing_bias;

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
        assert!(
            samples
                .last()
                .unwrap()
                .position
                .distance(Point::new(100.0, 0.0))
                < 3.0
        );
    }
}
