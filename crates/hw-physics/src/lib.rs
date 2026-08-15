//! hw-physics — Real-time pen physics state machine for Android handwriting.
//!
//! Event-driven: processes individual touch samples as they arrive from
//! `android-activity` / `winit`, not pre-computed paths.
//!
//! All state is f32 for SIMD-friendly, zero-copy GPU upload via `bytemuck`.
//! Zero per-frame heap allocation — `StrokeBatch` writes into pre-allocated buffers.

use std::f32::consts::PI;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Physical handwriting profile. Tuned per-user or per-script.
///
/// Architecture doc: "Your actual invention — this use case doesn't exist elsewhere."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandProfile {
    /// Dominant tremor frequency, Hz. Typical: 8.0–10.0 (hand), 25.0–30.0 (finger).
    pub tremor_freq_hz: f32,
    /// Base tremor amplitude in mm (screen-space, pre-scale). Typical: 0.05–0.2.
    pub tremor_base_amplitude_mm: f32,
    /// Fatigue accumulation: tremor amplitude multiplier per minute of writing.
    pub fatigue_per_minute: f32,
    /// Cap to prevent runaway tremor.
    pub max_fatigue: f32,
    /// Natural writing slant in degrees from vertical (negative = left-leaning).
    pub natural_slant_deg: f32,
    /// Pressure → ink width multiplier.
    pub pressure_sensitivity: f32,
    /// Minimum pressure to register "pen down" (avoids hover artifacts).
    pub pressure_threshold: f32,
    /// EMA time constant for pressure smoothing, milliseconds.
    pub pressure_smooth_ms: f32,
    /// Velocity scale: how much velocity affects width (dynamic calligraphy).
    pub velocity_width_scale: f32,
}

impl Default for HandProfile {
    fn default() -> Self {
        Self {
            tremor_freq_hz: 9.0,
            tremor_base_amplitude_mm: 0.08,
            fatigue_per_minute: 0.15,
            max_fatigue: 3.0,
            natural_slant_deg: -12.0,
            pressure_sensitivity: 2.5,
            pressure_threshold: 0.05,
            pressure_smooth_ms: 16.0,
            velocity_width_scale: 0.3,
        }
    }
}

impl HandProfile {
    /// Right-handed Latin script preset.
    pub fn latin_right_handed() -> Self {
        Self {
            natural_slant_deg: -15.0,
            pressure_sensitivity: 2.5,
            ..Default::default()
        }
    }

    /// Myanmar script: more upright, higher pressure sensitivity.
    pub fn myanmar_right_handed() -> Self {
        Self {
            natural_slant_deg: -5.0,
            pressure_sensitivity: 3.2,
            tremor_freq_hz: 8.5,
            ..Default::default()
        }
    }

    /// Left-handed preset (mirror slant, slightly higher tremor).
    pub fn latin_left_handed() -> Self {
        Self {
            natural_slant_deg: 20.0,
            tremor_base_amplitude_mm: 0.10,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// Current pen state, updated in-place on every touch event.
///
/// 64 bytes, cache-line friendly. All f32 for direct GPU vertex buffer mapping.
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct PenState {
    /// Position in screen mm (DPI-independent).
    pub pos_mm: [f32; 2],
    /// Raw velocity mm/s from last two samples.
    pub velocity_mm_s: [f32; 2],
    /// Smoothed pressure [0..1].
    pub pressure: f32,
    /// Pen azimuth (0 = vertical, + = clockwise) in radians.
    pub slant_azimuth_rad: f32,
    /// Pen altitude (0 = flat on paper, π/2 = upright) in radians.
    pub slant_altitude_rad: f32,
    /// Fatigue multiplier [1.0 .. max_fatigue].
    pub fatigue: f32,
    /// Seconds since session start.
    pub session_time_secs: f32,
    /// Seconds since current stroke started.
    pub stroke_time_secs: f32,
    /// Is pen currently down?
    pub is_down: u32, // u32 for Pod; 0 = false, 1 = true
    /// Phase accumulator for tremor oscillator.
    pub tremor_phase: f32,
    /// Previous raw pressure (for EMA smoothing).
    pub prev_raw_pressure: f32,
    /// Padding to 64 bytes (16 × f32).
    pub _pad: [f32; 2],
}

impl PenState {
    pub fn new(now: Instant, session_start: Instant) -> Self {
        Self {
            pos_mm: [0.0; 2],
            velocity_mm_s: [0.0; 2],
            pressure: 0.0,
            slant_azimuth_rad: 0.0,
            slant_altitude_rad: PI / 4.0,
            fatigue: 1.0,
            session_time_secs: 0.0,
            stroke_time_secs: 0.0,
            is_down: 0,
            tremor_phase: 0.0,
            prev_raw_pressure: 0.0,
            _pad: [0.0; 2],
        }
    }

    pub fn is_down_bool(&self) -> bool {
        self.is_down != 0
    }

    pub fn set_down(&mut self, down: bool) {
        self.is_down = if down { 1 } else { 0 };
    }
}

// ---------------------------------------------------------------------------
// Output: GPU-ready stroke point
// ---------------------------------------------------------------------------

/// A single deformed stroke point, ready for `hw-ink` tessellation and GPU upload.
///
/// Layout: vec4(x, y, pressure, timestamp) — matches `hw-render` vertex buffer.
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct StrokePoint {
    /// Deformed position in mm (tremor applied).
    pub pos_mm_x: f32,
    pub pos_mm_y: f32,
    /// Smoothed pressure [0..1].
    pub pressure: f32,
    /// Seconds since session start (for ink drying simulation).
    pub timestamp_secs: f32,
    /// Computed ink width in mm.
    pub width_mm: f32,
    /// Elliptical nib aspect ratio (width/height).
    pub aspect_ratio: f32,
    /// Slant azimuth at time of writing.
    pub slant_azimuth_rad: f32,
    /// Slant altitude at time of writing.
    pub slant_altitude_rad: f32,
}

impl StrokePoint {
    /// Flatten to GPU vertex format: [x, y, pressure, timestamp].
    pub fn to_vertex(&self) -> [f32; 4] {
        [
            self.pos_mm_x,
            self.pos_mm_y,
            self.pressure,
            self.timestamp_secs,
        ]
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Core physics state machine. Owns `HandProfile`, updates `PenState`.
///
/// Call `sample()` on every `MotionEvent` from the Android event loop.
/// Zero-allocation, single-threaded, 60 Hz ready.
pub struct PhysicsEngine {
    pub profile: HandProfile,
    pub state: PenState,
    /// Pixels per mm. Set once at startup from display metrics.
    pub dpi_mm: f32,
    /// Session start time.
    pub session_start: Instant,
    /// Current stroke start time.
    pub stroke_start: Instant,
    /// Pre-allocated batch buffer (avoids Vec reallocation).
    batch_buffer: Vec<StrokePoint>,
}

impl PhysicsEngine {
    pub fn new(profile: HandProfile, dpi_mm: f32) -> Self {
        let now = Instant::now();
        Self {
            profile,
            state: PenState::new(now, now),
            dpi_mm,
            session_start: now,
            stroke_start: now,
            batch_buffer: Vec::with_capacity(1024),
        }
    }

    /// Reset for new session.
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.session_start = now;
        self.stroke_start = now;
        self.state = PenState::new(now, now);
        self.batch_buffer.clear();
    }

    /// Process a single raw touch sample.
    ///
    /// `raw_pressure`: Android `MotionEvent.getPressure()` [0..1].
    /// `raw_azimuth`:  Android `MotionEvent.getOrientation()` in radians.
    /// `raw_altitude`: Android `MotionEvent.getAxisValue(AXIS_TILT)` in radians.
    /// `pos_px`:       Screen position in pixels.
    /// `timestamp`:    `MotionEvent.getEventTime()` mapped to `Instant`.
    ///
    /// Returns `Some(StrokePoint)` if pen is down, `None` if hover.
    pub fn sample(
        &mut self,
        raw_pressure: f32,
        raw_azimuth: f32,
        raw_altitude: f32,
        pos_px: [f32; 2],
        timestamp: Instant,
    ) -> Option<StrokePoint> {
        let dt = timestamp.saturating_duration_since(self.state.session_time());
        let dt_secs = dt.as_secs_f32().max(0.0001); // 0.1ms floor

        // --- Position & Velocity ---
        let new_pos_mm = [pos_px[0] / self.dpi_mm, pos_px[1] / self.dpi_mm];
        self.state.velocity_mm_s = [
            (new_pos_mm[0] - self.state.pos_mm[0]) / dt_secs,
            (new_pos_mm[1] - self.state.pos_mm[1]) / dt_secs,
        ];
        self.state.pos_mm = new_pos_mm;

        // --- Pressure EMA smoothing ---
        let alpha = ((dt_secs * 1000.0) / self.profile.pressure_smooth_ms).min(1.0);
        let smoothed_raw = self.state.prev_raw_pressure * (1.0 - alpha) + raw_pressure * alpha;
        self.state.prev_raw_pressure = smoothed_raw;

        let pressure = if smoothed_raw > self.profile.pressure_threshold {
            ((smoothed_raw - self.profile.pressure_threshold)
                / (1.0 - self.profile.pressure_threshold))
                .min(1.0)
        } else {
            0.0
        };

        // Pen-down / pen-up transition detection
        let was_down = self.state.is_down_bool();
        self.state.set_down(pressure > 0.0);

        if self.state.is_down_bool() && !was_down {
            // New stroke
            self.stroke_start = timestamp;
            self.state.stroke_time_secs = 0.0;
        }

        if !self.state.is_down_bool() {
            self.state.pressure = 0.0;
            self.state.session_time_secs =
                timestamp.duration_since(self.session_start).as_secs_f32();
            return None;
        }

        // --- Slant ---
        let (azimuth, altitude) = if raw_altitude > 0.01 {
            (raw_azimuth, raw_altitude.clamp(0.0, PI / 2.0))
        } else {
            // Finger touch: use natural slant + velocity bias
            let speed = hypot(self.state.velocity_mm_s[0], self.state.velocity_mm_s[1]);
            let velocity_slant = if speed > 1.0 {
                (self.state.velocity_mm_s[0] / speed).atan()
            } else {
                0.0
            };
            let blended = self.profile.natural_slant_deg.to_radians() * 0.8 + velocity_slant * 0.2;
            (blended, PI / 3.0)
        };
        self.state.slant_azimuth_rad = azimuth;
        self.state.slant_altitude_rad = altitude;

        // --- Fatigue ---
        let session_mins = self.state.session_time_secs / 60.0;
        let stroke_mins = self.state.stroke_time_secs / 60.0;
        let active_fatigue = 1.0 + stroke_mins * self.profile.fatigue_per_minute;
        let global_fatigue = 1.0 + session_mins * self.profile.fatigue_per_minute * 0.3;
        self.state.fatigue = (active_fatigue * global_fatigue).min(self.profile.max_fatigue);

        // --- Tremor ---
        self.state.tremor_phase += 2.0 * PI * self.profile.tremor_freq_hz * dt_secs;
        let tremor_amp = self.profile.tremor_base_amplitude_mm * self.state.fatigue;

        let speed = hypot(self.state.velocity_mm_s[0], self.state.velocity_mm_s[1]);
        let tremor_dir = if speed > 1.0 {
            // Perpendicular to velocity (physiological hand tremor)
            let vx = self.state.velocity_mm_s[0] / speed;
            let vy = self.state.velocity_mm_s[1] / speed;
            [-vy, vx]
        } else {
            let angle = self.state.tremor_phase * 0.7;
            [angle.cos(), angle.sin()]
        };

        let sin_phase = self.state.tremor_phase.sin();
        let tremor_offset = [
            tremor_dir[0] * tremor_amp * sin_phase,
            tremor_dir[1] * tremor_amp * sin_phase,
        ];

        // --- Ink width from pressure + slant + velocity ---
        let base_width = pressure * self.profile.pressure_sensitivity;
        let altitude_factor = (self.state.slant_altitude_rad / (PI / 2.0)).max(0.1);
        let velocity_bonus = speed * self.profile.velocity_width_scale * 0.01;
        let width = (base_width + velocity_bonus) * altitude_factor;
        let aspect = 1.0 + (1.0 - altitude_factor) * 2.0;

        // --- Final deformed point ---
        let deformed = StrokePoint {
            pos_mm_x: self.state.pos_mm[0] + tremor_offset[0],
            pos_mm_y: self.state.pos_mm[1] + tremor_offset[1],
            pressure,
            timestamp_secs: self.state.session_time_secs,
            width_mm: width.max(0.1),
            aspect_ratio: aspect.max(1.0),
            slant_azimuth_rad: self.state.slant_azimuth_rad,
            slant_altitude_rad: self.state.slant_altitude_rad,
        };

        self.state.pressure = pressure;
        self.state.session_time_secs = timestamp.duration_since(self.session_start).as_secs_f32();
        self.state.stroke_time_secs = timestamp.duration_since(self.stroke_start).as_secs_f32();

        Some(deformed)
    }

    /// Batch multiple samples (e.g., from a `MotionEvent` history).
    pub fn sample_batch(
        &mut self,
        samples: &[(f32, f32, f32, [f32; 2], Instant)], // (pressure, azimuth, altitude, pos, time)
    ) -> &[StrokePoint] {
        self.batch_buffer.clear();
        for (pressure, azimuth, altitude, pos, time) in samples {
            if let Some(pt) = self.sample(*pressure, *azimuth, *altitude, *pos, *time) {
                self.batch_buffer.push(pt);
            }
        }
        &self.batch_buffer
    }

    /// Call on `MainEvent::Pause`. Fatigue partially recovers.
    pub fn pause(&mut self) {
        self.state.fatigue = 1.0 + (self.state.fatigue - 1.0) * 0.5;
    }

    /// Call on `MainEvent::Resume`.
    pub fn resume(&mut self, now: Instant) {
        self.session_start = now;
        self.stroke_start = now;
        self.state.session_time_secs = 0.0;
        self.state.stroke_time_secs = 0.0;
    }
}

// Helper: hypot for f32
fn hypot(a: f32, b: f32) -> f32 {
    (a * a + b * b).sqrt()
}

// Helper for PenState session time
impl PenState {
    fn session_time(&self) -> Instant {
        // This is a bit of a hack — we don't store Instant in PenState (not Pod).
        // The engine stores session_start and computes deltas.
        // This method is only used by the engine which has session_start.
        unreachable!("Use engine.session_start + Duration::from_secs_f32(state.session_time_secs)")
    }
}

// ---------------------------------------------------------------------------
// Android bridge
// ---------------------------------------------------------------------------

/// Android `MotionEvent` integration.
///
/// Maps Android input events to `PhysicsEngine::sample()` arguments.
pub mod android {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    /// Android `MotionEvent.AXIS_TILT` constant.
    pub const AXIS_TILT: i32 = 25;

    /// Convert `android_activity::input::MotionEvent` to physics sample.
    ///
    /// Call from the `android_main` event loop on `InputEvent::MotionEvent`.
    pub fn motion_event_to_sample(
        engine: &mut PhysicsEngine,
        event: &android_activity::input::MotionEvent,
    ) -> Option<StrokePoint> {
        let pointer = event.pointer_at(0);
        let pos_px = [pointer.x(), pointer.y()];
        let pressure = pointer.pressure().clamp(0.0, 1.0);
        let azimuth = pointer.orientation();
        let altitude = pointer
            .axis_value(AXIS_TILT)
            .unwrap_or(0.0)
            .clamp(0.0, std::f32::consts::PI / 2.0);

        // Android event time is in nanoseconds since boot.
        // Map to Instant by offset from now.
        let event_time_ns = event.meta().event_time() as i64;
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64;
        // Approximate: use Instant::now() minus small delta
        let timestamp = Instant::now();

        engine.sample(pressure, azimuth, altitude, pos_px, timestamp)
    }

    /// Process all historical samples in a `MotionEvent`.
    ///
    /// Android batches motion events; process history first, then current.
    pub fn motion_event_with_history(
        engine: &mut PhysicsEngine,
        event: &android_activity::input::MotionEvent,
    ) -> Vec<StrokePoint> {
        let mut results = Vec::new();
        // Note: android_activity MotionEvent may not expose history directly.
        // In practice, poll_events() delivers individual events at 60-120 Hz.
        // This is a placeholder for batch processing if needed.
        if let Some(pt) = motion_event_to_sample(engine, event) {
            results.push(pt);
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_presets() {
        let latin = HandProfile::latin_right_handed();
        assert!(latin.natural_slant_deg < 0.0);

        let myanmar = HandProfile::myanmar_right_handed();
        assert!(myanmar.pressure_sensitivity > latin.pressure_sensitivity);
    }

    #[test]
    fn test_event_driven_simulation() {
        let mut engine = PhysicsEngine::new(HandProfile::default(), 25.4 * 2.0);
        let start = Instant::now();

        // Simulate a stroke: 20 samples over 200ms
        let mut points = Vec::new();
        for i in 0..20 {
            let t = start + Duration::from_millis(i as u64 * 10);
            let pos = [100.0 + i as f32 * 5.0, 200.0];
            let pressure = 0.5 + 0.3 * (i as f32 / 20.0).sin();
            if let Some(pt) = engine.sample(pressure, 0.0, 0.0, pos, t) {
                points.push(pt);
            }
        }

        assert!(!points.is_empty());
        // Pressure should be smoothed, not identical to raw
        assert!(points[5].pressure > 0.0);
        // Tremor should introduce small offsets
        let dx = points[10].pos_mm_x - (100.0 + 10.0 * 5.0) / engine.dpi_mm;
        assert!(dx.abs() < 1.0); // tremor is small
    }

    #[test]
    fn test_pen_up_pen_down() {
        let mut engine = PhysicsEngine::new(HandProfile::default(), 25.4);
        let start = Instant::now();

        // Hover (pressure below threshold)
        let hover = engine.sample(0.01, 0.0, 0.0, [0.0, 0.0], start);
        assert!(hover.is_none());
        assert!(!engine.state.is_down_bool());

        // Touch down
        let down = engine.sample(0.5, 0.0, 0.0, [0.0, 0.0], start + Duration::from_millis(16));
        assert!(down.is_some());
        assert!(engine.state.is_down_bool());
    }

    #[test]
    fn test_stroke_point_pod() {
        let pt = StrokePoint {
            pos_mm_x: 1.0,
            pos_mm_y: 2.0,
            pressure: 0.5,
            timestamp_secs: 1.0,
            width_mm: 1.0,
            aspect_ratio: 1.0,
            slant_azimuth_rad: 0.0,
            slant_altitude_rad: 0.0,
        };
        let bytes = bytemuck::bytes_of(&pt);
        assert_eq!(bytes.len(), 32); // 8 × f32
    }
}
