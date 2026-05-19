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

    /// Run a chat line that starts with `/`. The leading `/` is stripped
    /// before tokenisation. Returns the effects to enqueue. Returns
    /// `Err(msg)` for unknown commands or parse errors; caller logs that
    /// as a red line.
    pub fn dispatch(&self, line: &str) -> Result<Vec<UiEffect>, String> {
        let trimmed = line.trim();
        let body = trimmed.strip_prefix('/').unwrap_or(trimmed);
        let mut parts = body.split_whitespace();
        let name = parts.next().ok_or_else(|| "empty command".to_string())?;
        let args: Vec<&str> = parts.collect();
        let cmd = self.find(name).ok_or_else(|| format!("unknown command: /{name}"))?;
        cmd.run(&args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    impl Command for Echo {
        fn name(&self) -> &'static str { "echo" }
        fn help(&self) -> &'static str { "echo <text>" }
        fn run(&self, args: &[&str]) -> Result<Vec<UiEffect>, String> {
            Ok(vec![UiEffect::PostMessage(args.join(" "))])
        }
    }

    fn registry_with_echo() -> Registry {
        Registry { commands: vec![Box::new(Echo)] }
    }

    #[test]
    fn dispatch_known_command() {
        let r = registry_with_echo();
        let effs = r.dispatch("/echo hi there").unwrap();
        assert_eq!(effs, vec![UiEffect::PostMessage("hi there".into())]);
    }

    #[test]
    fn dispatch_unknown_returns_error() {
        let r = registry_with_echo();
        let err = r.dispatch("/nope").unwrap_err();
        assert!(err.contains("unknown"));
    }

    #[test]
    fn dispatch_strips_leading_slash_optional() {
        let r = registry_with_echo();
        let effs = r.dispatch("echo bare").unwrap();
        assert_eq!(effs, vec![UiEffect::PostMessage("bare".into())]);
    }
}
