//! Pause-menu items + activation. The menu is data-driven by a static
//! list so reordering or adding items is a one-line change.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MenuItem {
    Resume,
    SaveNow,
    Settings,
    Quit,
}

impl MenuItem {
    pub fn label(self) -> &'static str {
        match self {
            MenuItem::Resume   => "Resume",
            MenuItem::SaveNow  => "Save now",
            MenuItem::Settings => "Settings",
            MenuItem::Quit     => "Quit to desktop",
        }
    }
}

pub const TOP_MENU: &[MenuItem] = &[
    MenuItem::Resume,
    MenuItem::SaveNow,
    MenuItem::Settings,
    MenuItem::Quit,
];

/// Result of pressing/clicking a menu item. Decoupled from `UiEffect`
/// because some actions (Resume, Settings) only change UI state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MenuAction {
    Resume,
    Save,
    OpenSettings,
    BackToTop,
    Quit,
}
