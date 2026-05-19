//! Slash-command dispatcher. Each `Command` is a tiny struct that owns
//! its name + help string and turns `args: &[&str]` into a list of
//! `UiEffect`s (or a parse-error string the dispatcher will render as
//! a red `CommandError` line).

use crate::ui::effect::UiEffect;

pub trait Command {
    fn name(&self) -> &'static str;
    fn help(&self) -> &'static str;
    fn run(&self, args: &[&str]) -> Result<Vec<UiEffect>, String>;
}

pub struct Registry {
    commands: Vec<Box<dyn Command>>,
}

impl Registry {
    pub fn builtin() -> Self {
        Self { commands: vec![] } // populated in a later task
    }

    pub fn find(&self, name: &str) -> Option<&dyn Command> {
        self.commands.iter().find(|c| c.name() == name).map(|c| c.as_ref())
    }

    pub fn all(&self) -> &[Box<dyn Command>] {
        &self.commands
    }
}
