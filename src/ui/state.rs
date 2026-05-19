//! UI state machine. `UiState` carries the full state; `MenuNav` tracks
//! which pause-menu screen we're on and which item is hovered.

use crate::ui::chat::ChatInput;

#[derive(Debug)]
pub enum UiState {
    Playing,
    Paused { menu: MenuNav },
    Chat { input: ChatInput },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MenuNav {
    Top { hovered: usize },
    Settings,
}
