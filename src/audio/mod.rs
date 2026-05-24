//! Binary-side audio runtime.
//!
//! The library crate stays headless; this module owns the OS audio backend,
//! authored sound config, per-frame event routing, and the small bits of
//! gameplay state needed to avoid spammy playback.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use glam::Vec3;
use serde::Deserialize;

use crate::voxel::block::Block;

const DEFAULT_AUDIO_CONFIG: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/audio/default.ron");
const MUSIC_INITIAL_SILENCE: Duration = Duration::from_secs(90);
const MUSIC_MIN_SILENCE: Duration = Duration::from_secs(150);
const MUSIC_SILENCE_RANGE: Duration = Duration::from_secs(240);
const FOOTSTEP_FADE: Duration = Duration::from_millis(320);
const FOOTSTEP_RETRIGGER_FADE: Duration = Duration::from_millis(80);

#[derive(Debug, thiserror::Error)]
pub enum AudioConfigError {
    #[error("failed to read audio config {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to parse audio config {path}: {source}")]
    Parse {
        path: PathBuf,
        source: ron::error::SpannedError,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum AudioInitError {
    #[error("failed to start audio backend: {0}")]
    Backend(String),
    #[error("failed to create audio mixer track: {0}")]
    Track(String),
}

#[derive(Debug, thiserror::Error)]
pub enum AudioAssetError {
    #[error("audio asset `{id}` is missing at {path}")]
    Missing { id: String, path: PathBuf },
    #[error("audio asset `{id}` failed to decode from {path}: {reason}")]
    Decode {
        id: String,
        path: PathBuf,
        reason: String,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum AudioPlayError {
    #[error("audio event {event_kind} referenced unavailable sound `{sound_id}`")]
    UnavailableSound {
        event_kind: &'static str,
        sound_id: String,
    },
    #[error("audio event {event_kind} failed for sound `{sound_id}`: {reason}")]
    Playback {
        event_kind: &'static str,
        sound_id: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub master_volume: f32,
    pub music_volume: f32,
    pub ambient_volume: f32,
    pub sfx_volume: f32,
    pub music: MusicConfig,
    pub ui: UiAudioConfig,
    pub player: PlayerAudioConfig,
    pub ambient: AmbientConfig,
    #[serde(default)]
    pub mobs: Vec<MobSoundBankConfig>,
}

impl AudioConfig {
    pub fn load_default() -> Result<Self, AudioConfigError> {
        Self::load_from(DEFAULT_AUDIO_CONFIG)
    }

    pub fn load_from(path: impl AsRef<Path>) -> Result<Self, AudioConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| AudioConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text).map_err(|source| AudioConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn parse(text: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(text)
    }

    fn static_clips(&self) -> Vec<&AudioClip> {
        let mut clips = vec![&self.ui.click];
        if let Some(clip) = &self.player.jump {
            clips.push(clip);
        }
        if let Some(clip) = &self.player.land {
            clips.push(clip);
        }
        for bank in &self.player.footsteps {
            clips.extend(bank.clips.iter());
        }
        for bank in &self.mobs {
            for group in &bank.vocalizations {
                clips.extend(group.clips.iter());
            }
        }
        clips
    }

    fn base_path(path: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MusicConfig {
    pub playlist: Vec<AudioClip>,
    #[serde(default)]
    pub shuffle: bool,
    #[serde(default = "default_pause_music_volume")]
    pub pause_volume: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UiAudioConfig {
    pub click: AudioClip,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PlayerAudioConfig {
    #[serde(default)]
    pub jump: Option<AudioClip>,
    #[serde(default)]
    pub land: Option<AudioClip>,
    pub footsteps: Vec<FootstepBankConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FootstepBankConfig {
    pub id: String,
    pub material: String,
    pub blocks: Vec<Block>,
    pub clips: Vec<AudioClip>,
    #[serde(default)]
    pub sequential: bool,
    #[serde(default = "default_footstep_cooldown_ms")]
    pub cooldown_ms: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AmbientConfig {
    pub layers: Vec<AmbientLayerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AmbientLayerConfig {
    pub id: String,
    pub kind: AmbientLayerKind,
    pub clip: AudioClip,
    #[serde(default = "one")]
    pub base_volume: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum AmbientLayerKind {
    Wind,
    Day,
    Night,
    Water,
    Lava,
    Cave,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MobSoundBankConfig {
    pub mob_id: String,
    pub vocalizations: Vec<MobVocalizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MobVocalizationConfig {
    pub kind: String,
    pub clips: Vec<AudioClip>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioClip {
    pub id: String,
    pub path: String,
    #[serde(default = "one")]
    pub volume: f32,
}

#[derive(Debug, Clone)]
pub enum AudioEvent {
    UiClick,
    FootstepState {
        pos: Vec3,
        block: Option<Block>,
        horizontal_speed: f32,
        bob_phase: f32,
        walking: bool,
    },
    Jump,
    Land {
        impact: f32,
        block: Option<Block>,
    },
    MusicState {
        time_of_day: f32,
        paused: bool,
    },
    AmbientProbe(AmbientProbe),
    #[allow(dead_code)]
    MobVocalize {
        mob_id: String,
        pos: Vec3,
        kind: String,
    },
}

impl AudioEvent {
    fn kind(&self) -> &'static str {
        match self {
            AudioEvent::UiClick => "UiClick",
            AudioEvent::FootstepState { .. } => "FootstepState",
            AudioEvent::Jump => "Jump",
            AudioEvent::Land { .. } => "Land",
            AudioEvent::MusicState { .. } => "MusicState",
            AudioEvent::AmbientProbe(_) => "AmbientProbe",
            AudioEvent::MobVocalize { .. } => "MobVocalize",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AmbientProbe {
    pub listener_pos: Vec3,
    pub time_of_day: f32,
    pub nearby_water: f32,
    pub nearby_lava: f32,
    pub undergroundness: f32,
}

pub struct AudioEngine {
    config: AudioConfig,
    backend: AudioBackend,
}

enum AudioBackend {
    Kira(Box<KiraRuntime>),
    #[allow(dead_code)]
    Null(NullRuntime),
    Disabled {
        reason: String,
    },
}

#[derive(Default)]
struct NullRuntime {
    played: Vec<String>,
    ambient_targets: HashMap<String, f32>,
    footstep_cadence: FootstepCadence,
    footstep_sequence: HashMap<String, usize>,
}

impl AudioEngine {
    pub fn from_default_config() -> Result<Self, AudioConfigError> {
        let config = AudioConfig::load_default()?;
        Ok(Self::with_kira_or_disabled(config))
    }

    fn with_kira_or_disabled(config: AudioConfig) -> Self {
        match KiraRuntime::new(&config) {
            Ok(runtime) => Self {
                config,
                backend: AudioBackend::Kira(Box::new(runtime)),
            },
            Err(err) => {
                log::warn!("audio disabled: {err}");
                Self {
                    config,
                    backend: AudioBackend::Disabled {
                        reason: err.to_string(),
                    },
                }
            }
        }
    }

    #[cfg(test)]
    fn null(config: AudioConfig) -> Self {
        Self {
            config,
            backend: AudioBackend::Null(NullRuntime::default()),
        }
    }

    pub fn drain_events(&mut self, events: impl IntoIterator<Item = AudioEvent>) {
        for event in events {
            if let Err(err) = self.handle_event(&event) {
                log::warn!("{err}");
            }
        }
    }

    fn handle_event(&mut self, event: &AudioEvent) -> Result<(), AudioPlayError> {
        match &mut self.backend {
            AudioBackend::Kira(runtime) => runtime.handle_event(&self.config, event),
            AudioBackend::Null(runtime) => {
                handle_null_event(&self.config, runtime, event);
                Ok(())
            }
            AudioBackend::Disabled { reason } => {
                log::debug!(
                    "audio event {} ignored while disabled: {reason}",
                    event.kind()
                );
                Ok(())
            }
        }
    }

    #[cfg(test)]
    fn test_played_ids(&self) -> &[String] {
        match &self.backend {
            AudioBackend::Null(runtime) => &runtime.played,
            _ => &[],
        }
    }
}

type StaticHandle = kira::sound::static_sound::StaticSoundHandle;
type StreamingHandle = kira::sound::streaming::StreamingSoundHandle<kira::sound::FromFileError>;

struct KiraRuntime {
    _manager: kira::AudioManager<kira::DefaultBackend>,
    music_track: kira::track::TrackHandle,
    ambient_track: kira::track::TrackHandle,
    sfx_track: kira::track::TrackHandle,
    static_sounds: HashMap<String, kira::sound::static_sound::StaticSoundData>,
    ambient_layers: Vec<AmbientLayerRuntime>,
    music_handle: Option<StreamingHandle>,
    music_clip_volume: Option<f32>,
    next_music_at: Instant,
    music_generation: u64,
    active_footstep: Option<ActiveFootstep>,
    footstep_cadence: FootstepCadence,
    footstep_sequence: HashMap<String, usize>,
    next_footstep_at: Instant,
    unavailable: HashSet<String>,
}

struct AmbientLayerRuntime {
    id: String,
    kind: AmbientLayerKind,
    base_volume: f32,
    handle: Option<StreamingHandle>,
}

struct ActiveFootstep {
    bank_id: String,
    clip_id: String,
    handle: StaticHandle,
}

impl KiraRuntime {
    fn new(config: &AudioConfig) -> Result<Self, AudioInitError> {
        let mut manager =
            kira::AudioManager::<kira::DefaultBackend>::new(kira::AudioManagerSettings::default())
                .map_err(|err| AudioInitError::Backend(err.to_string()))?;
        let music_track = manager
            .add_sub_track(kira::track::TrackBuilder::new())
            .map_err(|err| AudioInitError::Track(err.to_string()))?;
        let ambient_track = manager
            .add_sub_track(kira::track::TrackBuilder::new())
            .map_err(|err| AudioInitError::Track(err.to_string()))?;
        let sfx_track = manager
            .add_sub_track(kira::track::TrackBuilder::new())
            .map_err(|err| AudioInitError::Track(err.to_string()))?;

        let mut runtime = Self {
            _manager: manager,
            music_track,
            ambient_track,
            sfx_track,
            static_sounds: HashMap::new(),
            ambient_layers: Vec::new(),
            music_handle: None,
            music_clip_volume: None,
            next_music_at: Instant::now() + MUSIC_INITIAL_SILENCE,
            music_generation: 0,
            active_footstep: None,
            footstep_cadence: FootstepCadence::default(),
            footstep_sequence: HashMap::new(),
            next_footstep_at: Instant::now(),
            unavailable: HashSet::new(),
        };
        runtime.load_static_sounds(config);
        runtime.start_ambient_layers(config);
        Ok(runtime)
    }

    fn load_static_sounds(&mut self, config: &AudioConfig) {
        for clip in config.static_clips() {
            if self.static_sounds.contains_key(&clip.id) || self.unavailable.contains(&clip.id) {
                continue;
            }
            let path = AudioConfig::base_path(&clip.path);
            if !path.exists() {
                let err = AudioAssetError::Missing {
                    id: clip.id.clone(),
                    path,
                };
                log::warn!("{err}");
                self.unavailable.insert(clip.id.clone());
                continue;
            }
            match kira::sound::static_sound::StaticSoundData::from_file(&path) {
                Ok(data) => {
                    self.static_sounds.insert(clip.id.clone(), data);
                }
                Err(err) => {
                    let err = AudioAssetError::Decode {
                        id: clip.id.clone(),
                        path,
                        reason: err.to_string(),
                    };
                    log::warn!("{err}");
                    self.unavailable.insert(clip.id.clone());
                }
            }
        }
    }

    fn start_ambient_layers(&mut self, config: &AudioConfig) {
        for layer in &config.ambient.layers {
            let path = AudioConfig::base_path(&layer.clip.path);
            let mut handle = None;
            if !path.exists() {
                let err = AudioAssetError::Missing {
                    id: layer.clip.id.clone(),
                    path,
                };
                log::warn!("{err}");
                self.unavailable.insert(layer.clip.id.clone());
            } else {
                let data =
                    kira::sound::streaming::StreamingSoundData::from_file(&path).map(|data| {
                        data.loop_region(..)
                            .volume(kira::Decibels::SILENCE)
                            .fade_in_tween(kira::Tween {
                                duration: Duration::from_millis(250),
                                ..Default::default()
                            })
                    });
                match data {
                    Ok(data) => match self.ambient_track.play(data) {
                        Ok(sound) => {
                            log::info!(
                                "audio start ambient layer={} kind={:?} clip={} path={}",
                                layer.id,
                                layer.kind,
                                layer.clip.id,
                                layer.clip.path
                            );
                            handle = Some(sound);
                        }
                        Err(err) => {
                            let err = AudioPlayError::Playback {
                                event_kind: "AmbientInit",
                                sound_id: layer.clip.id.clone(),
                                reason: err.to_string(),
                            };
                            log::warn!("{err}");
                        }
                    },
                    Err(err) => {
                        let err = AudioAssetError::Decode {
                            id: layer.clip.id.clone(),
                            path,
                            reason: err.to_string(),
                        };
                        log::warn!("{err}");
                        self.unavailable.insert(layer.clip.id.clone());
                    }
                }
            }
            self.ambient_layers.push(AmbientLayerRuntime {
                id: layer.id.clone(),
                kind: layer.kind,
                base_volume: layer.base_volume,
                handle,
            });
        }
    }

    fn update_music(&mut self, config: &AudioConfig, time_of_day: f32, paused: bool) {
        let now = Instant::now();
        if self
            .music_handle
            .as_ref()
            .is_some_and(|handle| handle.state() == kira::sound::PlaybackState::Stopped)
        {
            self.music_handle = None;
            self.music_clip_volume = None;
            self.next_music_at = now + self.next_music_silence();
        }
        if let Some(handle) = &mut self.music_handle {
            if let Some(clip_volume) = self.music_clip_volume {
                handle.set_volume(
                    gain_to_db(music_gain(config, clip_volume, paused)),
                    kira::Tween {
                        duration: Duration::from_millis(250),
                        ..Default::default()
                    },
                );
            }
            return;
        }
        if paused || now < self.next_music_at {
            return;
        }
        self.start_music(config, time_of_day, paused);
    }

    fn start_music(&mut self, config: &AudioConfig, time_of_day: f32, paused: bool) {
        let Some(clip) = current_music_clip(config, time_of_day, self.music_generation) else {
            return;
        };
        if self.unavailable.contains(&clip.id) {
            return;
        }
        let path = AudioConfig::base_path(&clip.path);
        if !path.exists() {
            let err = AudioAssetError::Missing {
                id: clip.id.clone(),
                path,
            };
            log::warn!("{err}");
            self.unavailable.insert(clip.id.clone());
            return;
        }
        let volume = music_gain(config, clip.volume, paused);
        match kira::sound::streaming::StreamingSoundData::from_file(&path).map(|data| {
            data.volume(gain_to_db(volume)).fade_in_tween(kira::Tween {
                duration: Duration::from_secs(2),
                ..Default::default()
            })
        }) {
            Ok(data) => match self.music_track.play(data) {
                Ok(handle) => {
                    log::info!(
                        "audio start music clip={} path={} time_of_day={time_of_day:.3}",
                        clip.id,
                        clip.path
                    );
                    self.music_handle = Some(handle);
                    self.music_clip_volume = Some(clip.volume);
                    self.music_generation = self.music_generation.wrapping_add(1);
                }
                Err(err) => {
                    let err = AudioPlayError::Playback {
                        event_kind: "MusicState",
                        sound_id: clip.id.clone(),
                        reason: err.to_string(),
                    };
                    log::warn!("{err}");
                }
            },
            Err(err) => {
                let err = AudioAssetError::Decode {
                    id: clip.id.clone(),
                    path,
                    reason: err.to_string(),
                };
                log::warn!("{err}");
                self.unavailable.insert(clip.id.clone());
            }
        }
    }

    fn next_music_silence(&self) -> Duration {
        let span = MUSIC_SILENCE_RANGE.as_secs();
        let offset = if span == 0 {
            0
        } else {
            self.music_generation.wrapping_mul(97) % span
        };
        MUSIC_MIN_SILENCE + Duration::from_secs(offset)
    }

    fn handle_event(
        &mut self,
        config: &AudioConfig,
        event: &AudioEvent,
    ) -> Result<(), AudioPlayError> {
        match event {
            AudioEvent::UiClick => {
                self.play_static(config, event.kind(), &config.ui.click, 1.0, 1.0)
            }
            AudioEvent::FootstepState {
                pos,
                block,
                horizontal_speed,
                bob_phase,
                walking,
            } => self.update_footstep_state(
                config,
                *pos,
                *block,
                *horizontal_speed,
                *bob_phase,
                *walking,
            ),
            AudioEvent::Jump => {
                if let Some(clip) = &config.player.jump {
                    self.play_static(config, event.kind(), clip, 1.0, 1.0)?;
                }
                Ok(())
            }
            AudioEvent::Land { impact, block } => {
                let gain = (impact.abs() / 12.0).clamp(0.3, 1.2);
                if let Some(block) = block
                    && let Some(bank) = footstep_bank(config, *block)
                    && let Some(clip) = bank.clips.first()
                {
                    log::info!(
                        "audio land source=footstep_bank bank={} material={} block={block:?} clip={}",
                        bank.id,
                        bank.material,
                        clip.id
                    );
                    self.play_static(config, event.kind(), clip, gain, 1.0)?;
                } else if let Some(clip) = &config.player.land {
                    log::info!(
                        "audio land source=fallback block={block:?} clip={}",
                        clip.id
                    );
                    self.play_static(config, event.kind(), clip, gain, 1.0)?;
                } else {
                    log::info!("audio land source=none block={block:?}");
                }
                Ok(())
            }
            AudioEvent::MusicState {
                time_of_day,
                paused,
            } => {
                self.update_music(config, *time_of_day, *paused);
                Ok(())
            }
            AudioEvent::AmbientProbe(probe) => {
                let targets = ambient_targets(&config.ambient.layers, *probe);
                for target in targets {
                    if let Some(layer) = self
                        .ambient_layers
                        .iter_mut()
                        .find(|l| l.id == target.id && l.kind == target.kind)
                        && let Some(handle) = &mut layer.handle
                    {
                        let gain = config.master_volume
                            * config.ambient_volume
                            * layer.base_volume
                            * target.volume;
                        handle.set_volume(
                            gain_to_db(gain),
                            kira::Tween {
                                duration: Duration::from_millis(300),
                                ..Default::default()
                            },
                        );
                    }
                }
                Ok(())
            }
            AudioEvent::MobVocalize { mob_id, pos, kind } => {
                let Some(bank) = config.mobs.iter().find(|bank| bank.mob_id == *mob_id) else {
                    return Ok(());
                };
                let Some(group) = bank.vocalizations.iter().find(|group| group.kind == *kind)
                else {
                    return Ok(());
                };
                if group.clips.is_empty() {
                    return Ok(());
                }
                let hash = event_hash(*pos, Block::Air, format!("{mob_id}:{kind}").as_bytes());
                let clip = &group.clips[(hash as usize) % group.clips.len()];
                self.play_static(config, event.kind(), clip, 1.0, 1.0)
            }
        }
    }

    fn update_footstep_state(
        &mut self,
        config: &AudioConfig,
        pos: Vec3,
        block: Option<Block>,
        horizontal_speed: f32,
        bob_phase: f32,
        walking: bool,
    ) -> Result<(), AudioPlayError> {
        let Some(block) = block.filter(|_| walking && horizontal_speed > 0.5) else {
            self.footstep_cadence.reset();
            self.stop_active_footstep();
            return Ok(());
        };
        let Some(bank) = footstep_bank(config, block) else {
            self.footstep_cadence.reset();
            self.stop_active_footstep();
            return Ok(());
        };
        if bank.clips.is_empty() {
            self.footstep_cadence.reset();
            self.stop_active_footstep();
            return Ok(());
        }
        if self
            .active_footstep
            .as_ref()
            .is_some_and(|active| active.bank_id != bank.id)
        {
            self.footstep_cadence.reset();
            self.stop_active_footstep();
        }
        let half_cycle = footstep_half_cycle(bob_phase);
        if !self
            .footstep_cadence
            .update(bob_phase, horizontal_speed, true, walking)
        {
            return Ok(());
        }
        let now = Instant::now();
        if now < self.next_footstep_at {
            return Ok(());
        }
        self.next_footstep_at = now + Duration::from_millis(bank.cooldown_ms);
        let hash = event_hash(pos, block, &half_cycle.to_le_bytes());
        let clip_index = if bank.sequential {
            next_sequence_index(&mut self.footstep_sequence, &bank.id, bank.clips.len())
        } else {
            (hash as usize) % bank.clips.len()
        };
        let clip = &bank.clips[clip_index];
        let rate = self.footstep_playback_rate(clip, horizontal_speed);
        let volume_jitter = 0.88 + normalized_hash(hash.rotate_left(17)) * 0.12;
        let gain = config.master_volume * config.sfx_volume * clip.volume * volume_jitter;
        self.stop_active_footstep_with_fade(FOOTSTEP_RETRIGGER_FADE);
        let Some(data) = self.static_sounds.get(&clip.id) else {
            return Err(AudioPlayError::UnavailableSound {
                event_kind: "FootstepState",
                sound_id: clip.id.clone(),
            });
        };
        let data = data
            .clone()
            .volume(gain_to_db(gain))
            .playback_rate(rate as f64)
            .fade_in_tween(kira::Tween {
                duration: Duration::from_millis(8),
                ..Default::default()
            });
        let handle = self
            .sfx_track
            .play(data)
            .map_err(|err| AudioPlayError::Playback {
                event_kind: "FootstepState",
                sound_id: clip.id.clone(),
                reason: err.to_string(),
            })?;
        log::info!(
            "audio play footstep bank={} material={} block={:?} clip={} path={} half_cycle={half_cycle} phase={bob_phase:.2} speed={horizontal_speed:.2} rate={rate:.2}",
            bank.id,
            bank.material,
            block,
            clip.id,
            clip.path
        );
        self.active_footstep = Some(ActiveFootstep {
            bank_id: bank.id.clone(),
            clip_id: clip.id.clone(),
            handle,
        });
        Ok(())
    }

    fn stop_active_footstep(&mut self) {
        self.stop_active_footstep_with_fade(FOOTSTEP_FADE);
    }

    fn stop_active_footstep_with_fade(&mut self, fade: Duration) {
        if let Some(mut active) = self.active_footstep.take() {
            log::info!(
                "audio stop footstep bank={} clip={} fade_ms={}",
                active.bank_id,
                active.clip_id,
                fade.as_millis()
            );
            active.handle.stop(kira::Tween {
                duration: fade,
                ..Default::default()
            });
        }
    }

    fn footstep_playback_rate(&self, _clip: &AudioClip, horizontal_speed: f32) -> f32 {
        footstep_playback_rate_for_speed(horizontal_speed)
    }

    fn play_static(
        &mut self,
        config: &AudioConfig,
        event_kind: &'static str,
        clip: &AudioClip,
        volume_multiplier: f32,
        playback_rate: f32,
    ) -> Result<(), AudioPlayError> {
        let Some(data) = self.static_sounds.get(&clip.id) else {
            return Err(AudioPlayError::UnavailableSound {
                event_kind,
                sound_id: clip.id.clone(),
            });
        };
        let gain = config.master_volume * config.sfx_volume * clip.volume * volume_multiplier;
        let data = data
            .clone()
            .volume(gain_to_db(gain))
            .playback_rate(playback_rate as f64);
        log::info!(
            "audio play event={event_kind} clip={} path={} rate={playback_rate:.2} gain={gain:.2}",
            clip.id,
            clip.path
        );
        self.sfx_track
            .play(data)
            .map(|_| ())
            .map_err(|err| AudioPlayError::Playback {
                event_kind,
                sound_id: clip.id.clone(),
                reason: err.to_string(),
            })
    }
}

fn handle_null_event(config: &AudioConfig, runtime: &mut NullRuntime, event: &AudioEvent) {
    match event {
        AudioEvent::UiClick => runtime.played.push(config.ui.click.id.clone()),
        AudioEvent::FootstepState {
            pos,
            block: Some(block),
            horizontal_speed,
            bob_phase,
            walking: true,
        } => {
            if *horizontal_speed > 0.5
                && let Some(bank) = footstep_bank(config, *block)
                && !bank.clips.is_empty()
                && runtime
                    .footstep_cadence
                    .update(*bob_phase, *horizontal_speed, true, true)
            {
                let half_cycle = footstep_half_cycle(*bob_phase);
                let hash = event_hash(*pos, *block, &half_cycle.to_le_bytes());
                let clip_index = if bank.sequential {
                    next_sequence_index(&mut runtime.footstep_sequence, &bank.id, bank.clips.len())
                } else {
                    (hash as usize) % bank.clips.len()
                };
                let clip = &bank.clips[clip_index];
                runtime.played.push(clip.id.clone());
            }
        }
        AudioEvent::FootstepState { .. } => runtime.footstep_cadence.reset(),
        AudioEvent::Jump => {
            if let Some(clip) = &config.player.jump {
                runtime.played.push(clip.id.clone());
            }
        }
        AudioEvent::Land { block, .. } => {
            let clip = block
                .and_then(|block| footstep_bank(config, block))
                .and_then(|bank| bank.clips.first())
                .or(config.player.land.as_ref());
            if let Some(clip) = clip {
                runtime.played.push(clip.id.clone());
            }
        }
        AudioEvent::AmbientProbe(probe) => {
            runtime.ambient_targets.clear();
            for target in ambient_targets(&config.ambient.layers, *probe) {
                runtime.ambient_targets.insert(target.id, target.volume);
            }
        }
        AudioEvent::MusicState { .. } | AudioEvent::MobVocalize { .. } => {}
    }
}

#[derive(Debug, Default)]
pub struct FootstepCadence {
    last_half_cycle: Option<u32>,
}

impl FootstepCadence {
    pub fn update(
        &mut self,
        bob_phase: f32,
        horizontal_speed: f32,
        grounded: bool,
        walking: bool,
    ) -> bool {
        if !grounded || !walking || horizontal_speed <= 0.5 {
            self.reset();
            return false;
        }
        let half_cycle = footstep_half_cycle(bob_phase);
        let triggered = self.last_half_cycle.is_none_or(|last| half_cycle > last);
        self.last_half_cycle = Some(half_cycle);
        triggered
    }

    pub fn reset(&mut self) {
        self.last_half_cycle = None;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AmbientLayerTarget {
    pub id: String,
    pub kind: AmbientLayerKind,
    pub volume: f32,
}

pub fn ambient_targets(
    layers: &[AmbientLayerConfig],
    probe: AmbientProbe,
) -> Vec<AmbientLayerTarget> {
    let daylight = daylight_factor(probe.time_of_day);
    let above_ground = 1.0 - probe.undergroundness.clamp(0.0, 1.0);
    layers
        .iter()
        .map(|layer| {
            let factor = match layer.kind {
                AmbientLayerKind::Wind => {
                    let height = ((probe.listener_pos.y - 72.0) / 80.0).clamp(0.0, 1.0);
                    (0.2 + 0.8 * height) * above_ground
                }
                AmbientLayerKind::Day => daylight * above_ground,
                AmbientLayerKind::Night => (1.0 - daylight) * above_ground,
                AmbientLayerKind::Water => probe.nearby_water.clamp(0.0, 1.0),
                AmbientLayerKind::Lava => probe.nearby_lava.clamp(0.0, 1.0),
                AmbientLayerKind::Cave => probe.undergroundness.clamp(0.0, 1.0),
            };
            AmbientLayerTarget {
                id: layer.id.clone(),
                kind: layer.kind,
                volume: (layer.base_volume * factor).clamp(0.0, 1.0),
            }
        })
        .collect()
}

pub fn footstep_bank(config: &AudioConfig, block: Block) -> Option<&FootstepBankConfig> {
    config
        .player
        .footsteps
        .iter()
        .find(|bank| bank.blocks.contains(&block))
}

fn current_music_clip(
    config: &AudioConfig,
    time_of_day: f32,
    generation: u64,
) -> Option<&AudioClip> {
    if config.music.playlist.is_empty() {
        return None;
    }
    let index = if config.music.shuffle {
        let slots = config.music.playlist.len();
        let day_slot = (time_of_day.rem_euclid(1.0) * slots as f32).floor() as usize;
        (day_slot + generation as usize) % slots
    } else {
        generation as usize % config.music.playlist.len()
    };
    config.music.playlist.get(index)
}

fn footstep_playback_rate_for_speed(horizontal_speed: f32) -> f32 {
    const NORMAL_WALK_SPEED: f32 = 5.0;
    (horizontal_speed / NORMAL_WALK_SPEED).clamp(0.55, 1.8)
}

fn footstep_half_cycle(bob_phase: f32) -> u32 {
    (bob_phase / std::f32::consts::PI).floor().max(0.0) as u32
}

fn next_sequence_index(
    sequences: &mut HashMap<String, usize>,
    bank_id: &str,
    clip_count: usize,
) -> usize {
    let next = sequences.entry(bank_id.to_owned()).or_default();
    let index = *next % clip_count;
    *next = index + 1;
    index
}

fn music_gain(config: &AudioConfig, clip_volume: f32, paused: bool) -> f32 {
    let pause = if paused {
        config.music.pause_volume
    } else {
        1.0
    };
    config.master_volume * config.music_volume * clip_volume * pause
}

fn gain_to_db(gain: f32) -> kira::Decibels {
    let gain = gain.clamp(0.0, 4.0);
    if gain <= 0.0 {
        kira::Decibels::SILENCE
    } else {
        kira::Decibels(20.0 * gain.log10())
    }
}

fn daylight_factor(time_of_day: f32) -> f32 {
    let t = time_of_day.rem_euclid(1.0);
    (((t - 0.5) * std::f32::consts::TAU).cos() * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn event_hash(pos: Vec3, block: Block, salt: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in salt {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    for value in [
        pos.x.floor() as i32 as u32,
        pos.y.floor() as i32 as u32,
        pos.z.floor() as i32 as u32,
        block as u32,
    ] {
        hash ^= value as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn normalized_hash(hash: u64) -> f32 {
    ((hash >> 40) as u32 as f32) / ((1_u32 << 24) as f32)
}

fn one() -> f32 {
    1.0
}

fn default_pause_music_volume() -> f32 {
    0.45
}

fn default_footstep_cooldown_ms() -> u64 {
    90
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_CONFIG: &str = r#"
(
    master_volume: 0.8,
    music_volume: 0.6,
    ambient_volume: 0.7,
    sfx_volume: 0.9,
    music: (
        playlist: [(id: "menu_theme", path: "assets/audio/music/menu_theme.ogg", volume: 1.0)],
        shuffle: false,
        pause_volume: 0.4,
    ),
    ui: (
        click: (id: "ui_click", path: "assets/audio/ui/click.ogg", volume: 0.7),
    ),
    player: (
        jump: Some((id: "player_jump", path: "assets/audio/player/jump.ogg", volume: 0.8)),
        land: Some((id: "player_land", path: "assets/audio/player/land.ogg", volume: 0.9)),
        footsteps: [
            (
                id: "stone_steps",
                material: "stone",
                blocks: [Stone],
                clips: [
                    (id: "stone_step_1", path: "assets/audio/player/footsteps/stone_1.ogg", volume: 0.8),
                    (id: "stone_step_2", path: "assets/audio/player/footsteps/stone_2.ogg", volume: 0.8),
                ],
                sequential: true,
                cooldown_ms: 80,
            ),
            (
                id: "soil_steps",
                material: "soil",
                blocks: [Dirt, Grass],
                clips: [(id: "soil_step_1", path: "assets/audio/player/footsteps/soil_1.ogg", volume: 0.8)],
                cooldown_ms: 80,
            ),
        ],
    ),
    ambient: (
        layers: [
            (id: "day", kind: Day, clip: (id: "amb_day", path: "assets/audio/ambient/day.ogg", volume: 1.0), base_volume: 0.5),
            (id: "night", kind: Night, clip: (id: "amb_night", path: "assets/audio/ambient/night.ogg", volume: 1.0), base_volume: 0.5),
            (id: "cave", kind: Cave, clip: (id: "amb_cave", path: "assets/audio/ambient/cave.ogg", volume: 1.0), base_volume: 0.8),
            (id: "water", kind: Water, clip: (id: "amb_water", path: "assets/audio/ambient/water.ogg", volume: 1.0), base_volume: 0.7),
            (id: "lava", kind: Lava, clip: (id: "amb_lava", path: "assets/audio/ambient/lava.ogg", volume: 1.0), base_volume: 0.7),
        ],
    ),
    mobs: [
        (
            mob_id: "future_cow",
            vocalizations: [
                (kind: "idle", clips: [(id: "cow_idle_1", path: "assets/audio/mobs/future_cow/idle_1.ogg", volume: 1.0)]),
            ],
        ),
    ],
)
"#;

    #[test]
    fn parses_valid_config() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        assert_eq!(cfg.player.footsteps.len(), 2);
        assert_eq!(cfg.ambient.layers.len(), 5);
        assert_eq!(cfg.mobs[0].mob_id, "future_cow");
    }

    #[test]
    fn bundled_default_loads_and_parses() {
        let cfg = AudioConfig::load_default().unwrap();
        assert!(!cfg.player.footsteps.is_empty());
        assert!(!cfg.ambient.layers.is_empty());
    }

    #[test]
    fn bundled_default_references_existing_assets() {
        let cfg = AudioConfig::load_default().unwrap();
        let mut clips = cfg.static_clips();
        clips.extend(cfg.music.playlist.iter());
        for layer in &cfg.ambient.layers {
            clips.push(&layer.clip);
        }
        for clip in clips {
            assert!(
                AudioConfig::base_path(&clip.path).exists(),
                "missing audio asset `{}` at {}",
                clip.id,
                clip.path
            );
        }
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(AudioConfig::parse("(master_volume: \"loud\")").is_err());
    }

    #[test]
    fn footstep_cadence_triggers_on_half_cycle_crossing() {
        let mut cadence = FootstepCadence::default();
        assert!(cadence.update(0.2, 4.0, true, true));
        assert!(!cadence.update(std::f32::consts::PI * 0.9, 4.0, true, true));
        assert!(cadence.update(std::f32::consts::PI * 1.05, 4.0, true, true));
        assert!(cadence.update(std::f32::consts::PI * 2.05, 4.0, true, true));
        assert!(!cadence.update(std::f32::consts::PI * 1.1, 4.0, false, true));
        assert!(cadence.update(std::f32::consts::PI * 1.2, 4.0, true, true));
    }

    #[test]
    fn maps_block_to_footstep_bank() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        assert_eq!(footstep_bank(&cfg, Block::Stone).unwrap().id, "stone_steps");
        assert_eq!(footstep_bank(&cfg, Block::Grass).unwrap().id, "soil_steps");
        assert!(footstep_bank(&cfg, Block::Water).is_none());
    }

    #[test]
    fn ambient_targets_follow_day_night_cave_and_fluids() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        let day = ambient_targets(
            &cfg.ambient.layers,
            AmbientProbe {
                listener_pos: Vec3::new(0.0, 90.0, 0.0),
                time_of_day: 0.5,
                nearby_water: 0.0,
                nearby_lava: 0.0,
                undergroundness: 0.0,
            },
        );
        let night = ambient_targets(
            &cfg.ambient.layers,
            AmbientProbe {
                listener_pos: Vec3::new(0.0, 90.0, 0.0),
                time_of_day: 0.0,
                nearby_water: 0.6,
                nearby_lava: 0.3,
                undergroundness: 0.75,
            },
        );
        let day_volume = day.iter().find(|t| t.id == "day").unwrap().volume;
        let night_volume = night.iter().find(|t| t.id == "night").unwrap().volume;
        let cave_volume = night.iter().find(|t| t.id == "cave").unwrap().volume;
        let water_volume = night.iter().find(|t| t.id == "water").unwrap().volume;
        let lava_volume = night.iter().find(|t| t.id == "lava").unwrap().volume;
        assert!(day_volume > 0.45);
        assert!(night_volume > 0.1);
        assert!(cave_volume > 0.55);
        assert!(water_volume > 0.35);
        assert!(lava_volume > 0.2);
    }

    #[test]
    fn null_backend_routes_ui_and_footstep_events() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        let mut engine = AudioEngine::null(cfg);
        engine.drain_events([
            AudioEvent::UiClick,
            AudioEvent::FootstepState {
                pos: Vec3::new(1.0, 64.0, 1.0),
                block: Some(Block::Stone),
                horizontal_speed: 5.0,
                bob_phase: 0.0,
                walking: true,
            },
        ]);
        assert_eq!(engine.test_played_ids().len(), 2);
        assert_eq!(engine.test_played_ids()[0], "ui_click");
        assert!(engine.test_played_ids()[1].starts_with("stone_step_"));
    }

    #[test]
    fn sequential_footstep_bank_advances_in_clip_order() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        let mut engine = AudioEngine::null(cfg);
        engine.drain_events([
            AudioEvent::FootstepState {
                pos: Vec3::new(1.0, 64.0, 1.0),
                block: Some(Block::Stone),
                horizontal_speed: 5.0,
                bob_phase: 0.0,
                walking: true,
            },
            AudioEvent::FootstepState {
                pos: Vec3::new(1.0, 64.0, 1.0),
                block: Some(Block::Stone),
                horizontal_speed: 5.0,
                bob_phase: std::f32::consts::PI * 1.1,
                walking: true,
            },
            AudioEvent::FootstepState {
                pos: Vec3::new(1.0, 64.0, 1.0),
                block: Some(Block::Stone),
                horizontal_speed: 5.0,
                bob_phase: std::f32::consts::PI * 2.1,
                walking: true,
            },
        ]);
        assert_eq!(
            engine.test_played_ids(),
            &[
                "stone_step_1".to_string(),
                "stone_step_2".to_string(),
                "stone_step_1".to_string(),
            ]
        );
    }

    #[test]
    fn landing_uses_surface_material_bank() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        let mut engine = AudioEngine::null(cfg);
        engine.drain_events([AudioEvent::Land {
            impact: 4.0,
            block: Some(Block::Grass),
        }]);
        assert_eq!(engine.test_played_ids(), &["soil_step_1".to_string()]);
    }

    #[test]
    fn footstep_playback_rate_scales_with_speed() {
        assert!(footstep_playback_rate_for_speed(8.0) > footstep_playback_rate_for_speed(4.0));
        assert_eq!(footstep_playback_rate_for_speed(5.0), 1.0);
    }

    #[test]
    fn music_selection_advances_without_immediate_loop_bias() {
        let cfg = AudioConfig::parse(VALID_CONFIG).unwrap();
        let first = current_music_clip(&cfg, 0.5, 0).unwrap().id.as_str();
        let second = current_music_clip(&cfg, 0.5, 1).unwrap().id.as_str();
        assert_eq!(first, second);
    }
}
