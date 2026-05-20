//! A "session": one Generator + cameras + chunk cache wired up.
//! PR 4 will let the AppState hold two sessions for A/B compare.

use crate::camera::{Camera, FlyCamera, OrbitCamera};
use crate::crosssection::CrossSection;
use crate::overlays::MapView;
use crate::paint::PaintMode;
use crate::probe::Probe;
use crate::world::invalidate::Invalidator;
use crate::world::{Region, World};
use oxium::worldgen::config::{ConfigHolder, WorldgenConfig};
use oxium::worldgen::Generator;
use std::sync::Arc;

pub enum CamKind {
    Fly,
    Orbit,
}

pub struct Session {
    pub seed: u64,
    pub config: ConfigHolder,
    pub generator: Arc<Generator>,
    pub world: World,
    pub fly: FlyCamera,
    pub orbit: OrbitCamera,
    pub cam_kind: CamKind,
    pub paint: PaintMode,
    pub invalidator: Invalidator,
    pub probe: Probe,
    pub map: MapView,
    pub cross: CrossSection,
}

impl Session {
    pub fn new(seed: u64, config: WorldgenConfig, region: Region) -> Self {
        let holder = ConfigHolder::new(config);
        let generator = Arc::new(Generator::with_config(seed, holder.clone()));
        let world = World::new(generator.clone(), region);
        Self {
            seed,
            config: holder,
            generator,
            world,
            fly: FlyCamera::new(),
            orbit: OrbitCamera::new(),
            cam_kind: CamKind::Fly,
            paint: PaintMode::default(),
            invalidator: Invalidator::new(),
            probe: Probe::new(),
            map: MapView::new(),
            cross: CrossSection::new(),
        }
    }

    pub fn camera(&self) -> &dyn Camera {
        match self.cam_kind {
            CamKind::Fly => &self.fly,
            CamKind::Orbit => &self.orbit,
        }
    }

    pub fn toggle_camera(&mut self) {
        self.cam_kind = match self.cam_kind {
            CamKind::Fly => CamKind::Orbit,
            CamKind::Orbit => CamKind::Fly,
        };
    }
}
