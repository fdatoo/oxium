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
        Self {
            commands: vec![
                Box::new(CmdTp),
                Box::new(CmdTime),
                Box::new(CmdFly),
                Box::new(CmdSave),
                Box::new(CmdHelp),
                Box::new(CmdClear),
            ],
        }
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

struct CmdTp;
impl Command for CmdTp {
    fn name(&self) -> &'static str { "tp" }
    fn help(&self) -> &'static str { "/tp <x> <y> <z> — teleport the player" }
    fn run(&self, args: &[&str]) -> Result<Vec<UiEffect>, String> {
        if args.len() != 3 {
            return Err("usage: /tp <x> <y> <z>".into());
        }
        let parse = |s: &str| s.parse::<f32>().map_err(|_| format!("not a number: {s}"));
        let x = parse(args[0])?;
        let y = parse(args[1])?;
        let z = parse(args[2])?;
        Ok(vec![UiEffect::Teleport(glam::Vec3::new(x, y, z))])
    }
}

struct CmdTime;
impl Command for CmdTime {
    fn name(&self) -> &'static str { "time" }
    fn help(&self) -> &'static str { "/time <0..1> — set time of day" }
    fn run(&self, args: &[&str]) -> Result<Vec<UiEffect>, String> {
        let arg = args.first().ok_or("usage: /time <0..1>")?;
        let t = arg.parse::<f32>().map_err(|_| format!("not a number: {arg}"))?;
        Ok(vec![UiEffect::SetTime(t.clamp(0.0, 1.0))])
    }
}

struct CmdFly;
impl Command for CmdFly {
    fn name(&self) -> &'static str { "fly" }
    fn help(&self) -> &'static str { "/fly — toggle fly mode" }
    fn run(&self, _args: &[&str]) -> Result<Vec<UiEffect>, String> {
        Ok(vec![UiEffect::ToggleFly])
    }
}

struct CmdSave;
impl Command for CmdSave {
    fn name(&self) -> &'static str { "save" }
    fn help(&self) -> &'static str { "/save — force an autosave now" }
    fn run(&self, _args: &[&str]) -> Result<Vec<UiEffect>, String> {
        Ok(vec![
            UiEffect::Save,
            UiEffect::PostMessage("Saved.".into()),
        ])
    }
}

struct CmdHelp;
impl Command for CmdHelp {
    fn name(&self) -> &'static str { "help" }
    fn help(&self) -> &'static str { "/help — list commands" }
    fn run(&self, _args: &[&str]) -> Result<Vec<UiEffect>, String> {
        // /help is handled specially in submit_chat where the registry is
        // available; never reach this body.
        Err("__help_handled_by_caller__".into())
    }
}

struct CmdClear;
impl Command for CmdClear {
    fn name(&self) -> &'static str { "clear" }
    fn help(&self) -> &'static str { "/clear — clear the chat log" }
    fn run(&self, _args: &[&str]) -> Result<Vec<UiEffect>, String> {
        Ok(vec![UiEffect::ClearChat])
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

    #[test]
    fn cmd_tp_parses_three_floats() {
        let r = Registry::builtin();
        let effs = r.dispatch("/tp 1.5 64 -32").unwrap();
        assert_eq!(effs, vec![UiEffect::Teleport(glam::Vec3::new(1.5, 64.0, -32.0))]);
    }

    #[test]
    fn cmd_tp_rejects_bad_args() {
        let r = Registry::builtin();
        assert!(r.dispatch("/tp 1 2 abc").is_err());
        assert!(r.dispatch("/tp 1 2").is_err());
    }

    #[test]
    fn cmd_time_clamps() {
        let r = Registry::builtin();
        let effs = r.dispatch("/time -0.1").unwrap();
        assert_eq!(effs, vec![UiEffect::SetTime(0.0)]);
        let effs = r.dispatch("/time 1.5").unwrap();
        assert_eq!(effs, vec![UiEffect::SetTime(1.0)]);
    }

    #[test]
    fn cmd_fly_emits_toggle() {
        let r = Registry::builtin();
        let effs = r.dispatch("/fly").unwrap();
        assert_eq!(effs, vec![UiEffect::ToggleFly]);
    }

    #[test]
    fn cmd_clear_emits_clear() {
        let r = Registry::builtin();
        let effs = r.dispatch("/clear").unwrap();
        assert_eq!(effs, vec![UiEffect::ClearChat]);
    }

    #[test]
    fn cmd_save_emits_save_and_message() {
        let r = Registry::builtin();
        let effs = r.dispatch("/save").unwrap();
        assert!(effs.iter().any(|e| matches!(e, UiEffect::Save)));
        assert!(effs.iter().any(|e| matches!(e, UiEffect::PostMessage(_))));
    }
}
