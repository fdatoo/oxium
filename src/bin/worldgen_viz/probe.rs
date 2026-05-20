//! Column probe state: pinned (wx, wz), last `ColumnProbe` snapshot,
//! and the Y at which the user is currently inspecting density.

use oxium::worldgen::probe::{ColumnProbe, DensityBreakdown};
use oxium::worldgen::Generator;

pub struct Probe {
    pub pinned: Option<(i32, i32)>,
    pub snapshot: Option<ColumnProbe>,
    pub probe_y: i32,
    pub breakdown: Option<DensityBreakdown>,
}

impl Probe {
    pub fn new() -> Self {
        Self { pinned: None, snapshot: None, probe_y: 70, breakdown: None }
    }

    /// Pin a new column and refresh its snapshot.
    pub fn pin(&mut self, generator: &Generator, wx: i32, wz: i32) {
        let snapshot = generator.probe_column(wx, wz);
        self.probe_y = snapshot.h_target;
        self.snapshot = Some(snapshot.clone());
        self.pinned = Some((wx, wz));
        self.breakdown = Some(generator.evaluate_density_breakdown(wx, self.probe_y, wz));
    }

    pub fn unpin(&mut self) {
        self.pinned = None;
        self.snapshot = None;
        self.breakdown = None;
    }

    /// Refresh the snapshot + breakdown for the currently pinned
    /// column. Used when the config changes.
    pub fn refresh(&mut self, generator: &Generator) {
        if let Some((wx, wz)) = self.pinned {
            let snapshot = generator.probe_column(wx, wz);
            self.breakdown = Some(generator.evaluate_density_breakdown(wx, self.probe_y, wz));
            self.snapshot = Some(snapshot);
        }
    }

    /// Update the Y slider; refreshes the density breakdown only.
    pub fn set_y(&mut self, generator: &Generator, y: i32) {
        if y != self.probe_y {
            self.probe_y = y;
            if let Some((wx, wz)) = self.pinned {
                self.breakdown = Some(generator.evaluate_density_breakdown(wx, y, wz));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxium::worldgen::Generator;

    #[test]
    fn pin_populates_snapshot_and_breakdown() {
        let g = Generator::new(42);
        let mut p = Probe::new();
        p.pin(&g, 50, 50);
        assert_eq!(p.pinned, Some((50, 50)));
        assert!(p.snapshot.is_some());
        assert!(p.breakdown.is_some());
        assert_eq!(p.probe_y, p.snapshot.as_ref().unwrap().h_target);
    }

    #[test]
    fn unpin_clears_state() {
        let g = Generator::new(42);
        let mut p = Probe::new();
        p.pin(&g, 0, 0);
        p.unpin();
        assert!(p.pinned.is_none());
        assert!(p.snapshot.is_none());
        assert!(p.breakdown.is_none());
    }

    #[test]
    fn set_y_refreshes_breakdown_only() {
        let g = Generator::new(42);
        let mut p = Probe::new();
        p.pin(&g, 0, 0);
        let initial_breakdown = p.breakdown.unwrap();
        p.set_y(&g, p.probe_y + 20);
        let new_breakdown = p.breakdown.unwrap();
        // Y should have changed; breakdown's bias should differ.
        assert_ne!(initial_breakdown.bias.to_bits(), new_breakdown.bias.to_bits());
    }
}
