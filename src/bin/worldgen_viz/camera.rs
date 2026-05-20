//! Cameras for the viz: orbit (default) + fly (free-look).

use glam::{Mat4, Vec3};

/// What the renderer needs from any camera: a view+projection matrix.
pub trait Camera {
    fn view_proj(&self, aspect: f32) -> Mat4;
    fn position(&self) -> Vec3;
}

#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: Vec3::new(32.0, 70.0, 32.0),
            yaw: 0.8,
            pitch: 0.4,
            distance: 160.0,
        }
    }

    /// Pan the orbit target perpendicular to view. Scales with distance
    /// so the gesture feels the same whether zoomed in or out.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let right = Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos());
        let factor = self.distance * 0.0015;
        self.target -= right * dx * factor;
        self.target += Vec3::Y * dy * factor;
    }

    /// Re-center the orbit on `target`, keeping the current yaw / pitch / distance.
    pub fn focus_on(&mut self, target: Vec3) {
        self.target = target;
    }
}

impl Camera for OrbitCamera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), aspect, 0.5, 4096.0);
        proj * view
    }

    fn position(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.yaw.cos() * self.pitch.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.yaw.sin() * self.pitch.cos(),
            )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
}

impl FlyCamera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 96.0, 64.0),
            yaw: -1.0,
            pitch: -0.3,
            speed: 30.0,
        }
    }

    /// Forward unit vector in world space (XZ-yaw + Y-pitch).
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Right-hand vector orthogonal to forward in the XZ plane.
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, -self.yaw.cos()).normalize()
    }

    /// Apply WASD/QE motion. `forward_back` is +1 for W, −1 for S; `strafe`
    /// is +1 for D, −1 for A; `vertical` is +1 for E, −1 for Q. `dt` in
    /// seconds. Uses `self.speed` as units per second, boosted if `boost`.
    pub fn translate(&mut self, forward_back: f32, strafe: f32, vertical: f32, boost: bool, dt: f32) {
        let mul = if boost { 4.0 } else { 1.0 };
        // W/S walk along the yaw-only horizontal direction — pitch
        // controls where you're looking, Q/E controls elevation.
        // Mixing pitch into W/S means tilting the camera up also
        // climbs the camera, which fights the dedicated Q/E.
        let fwd_flat = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize();
        let v = fwd_flat * forward_back + self.right() * strafe + Vec3::Y * vertical;
        if v.length_squared() > 0.0 {
            self.position += v.normalize() * self.speed * mul * dt;
        }
    }

    /// Apply mouse delta. `dx`, `dy` are screen pixels.
    pub fn look(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch - dy * 0.005).clamp(-1.5, 1.5);
    }

    /// Translate the camera parallel to the view plane: along the
    /// camera's right vector for `dx` (horizontal), along world-up for
    /// `dy` (vertical). World-up rather than camera-up so panning
    /// doesn't drift the elevation as pitch changes — matches the
    /// behaviour every DCC editor's middle-mouse pan uses. Sign
    /// convention: drag right → world scrolls right (camera moves left).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let factor = 0.2;
        self.position -= self.right() * dx * factor;
        self.position += Vec3::Y * dy * factor;
    }

    /// Dolly along the forward direction. Positive `amount` moves forward.
    pub fn dolly(&mut self, amount: f32) {
        self.position += self.forward() * amount;
    }

    /// Reposition the camera near `target` and orient toward it.
    /// Camera ends up ~30 blocks back along the current XZ heading,
    /// 20 above the target — close enough to inspect, not so close
    /// that the column fills the screen.
    pub fn focus_on(&mut self, target: Vec3) {
        // Keep the current yaw heading; just place the camera so the
        // target sits along its forward at ~36 blocks distance.
        let dist = 36.0;
        let height_above = 20.0;
        let xz_dir = Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize();
        self.position = target - xz_dir * dist + Vec3::Y * height_above;
        let to_target = (target - self.position).normalize();
        self.pitch = to_target.y.asin().clamp(-1.5, 1.5);
        self.yaw = to_target.z.atan2(to_target.x);
    }
}

impl Camera for FlyCamera {
    fn view_proj(&self, aspect: f32) -> Mat4 {
        let target = self.position + self.forward();
        let view = Mat4::look_at_rh(self.position, target, Vec3::Y);
        let proj = Mat4::perspective_rh(60f32.to_radians(), aspect, 0.5, 4096.0);
        proj * view
    }

    fn position(&self) -> Vec3 {
        self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_translate_forward_moves_along_forward() {
        let mut cam = FlyCamera {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 10.0,
        };
        cam.translate(1.0, 0.0, 0.0, false, 1.0);
        let expected = cam.forward() * 10.0;
        assert!((cam.position - expected).length() < 1e-4, "got {:?}", cam.position);
    }

    #[test]
    fn fly_strafe_is_orthogonal_to_forward() {
        let cam = FlyCamera {
            position: Vec3::ZERO,
            yaw: 0.5,
            pitch: 0.0,
            speed: 1.0,
        };
        let dot = cam.forward().dot(cam.right());
        assert!(dot.abs() < 1e-4, "forward and right must be orthogonal, got dot={dot}");
    }

    #[test]
    fn fly_pitch_clamps_to_pi_over_two() {
        let mut cam = FlyCamera::new();
        // 100 frames of looking straight up at high dy.
        for _ in 0..100 {
            cam.look(0.0, -10000.0);
        }
        assert!(cam.pitch <= 1.5 && cam.pitch >= -1.5, "pitch escape: {}", cam.pitch);
    }

    #[test]
    fn orbit_position_distance_matches_field() {
        let cam = OrbitCamera::new();
        let d = (cam.position() - cam.target).length();
        assert!((d - cam.distance).abs() < 1e-3, "got distance {d}");
    }
}
