//! Slash-command registry backed by the pure command tree parser.

use std::sync::{Arc, OnceLock};

use crate::command::{
    CommandDispatcher, CommandError, CommandHint, CommandNode, CompletionList, F32Argument,
};
use crate::ui::effect::UiEffect;

#[derive(Debug, Default)]
pub struct UiCommandSource;

pub struct Registry {
    dispatcher: CommandDispatcher<UiCommandSource, Vec<UiEffect>>,
    source: UiCommandSource,
}

impl Registry {
    pub fn builtin() -> Self {
        let source = UiCommandSource;
        let mut dispatcher = CommandDispatcher::new();
        let help_lines: Arc<OnceLock<Vec<String>>> = Arc::new(OnceLock::new());

        dispatcher.register(
            CommandNode::literal("tp").then(
                CommandNode::argument("x", F32Argument).then(
                    CommandNode::argument("y", F32Argument).then(
                        CommandNode::argument("z", F32Argument)
                            .help("teleport the player")
                            .executes(|ctx| {
                                let x = ctx.f32("x").expect("x is parsed by F32Argument");
                                let y = ctx.f32("y").expect("y is parsed by F32Argument");
                                let z = ctx.f32("z").expect("z is parsed by F32Argument");
                                vec![UiEffect::Teleport(glam::Vec3::new(x, y, z))]
                            }),
                    ),
                ),
            ),
        );
        dispatcher.register(
            CommandNode::literal("teleport")
                .redirects_to(["tp"])
                .help("alias for /tp"),
        );
        dispatcher.register(
            CommandNode::literal("time").then(
                CommandNode::argument("t", F32Argument)
                    .help("set time of day")
                    .executes(|ctx| {
                        let t = ctx.f32("t").expect("t is parsed by F32Argument");
                        vec![UiEffect::SetTime(t.clamp(0.0, 1.0))]
                    }),
            ),
        );
        dispatcher.register(
            CommandNode::literal("fly")
                .help("toggle fly mode")
                .executes(|_| vec![UiEffect::ToggleFly]),
        );
        dispatcher.register(
            CommandNode::literal("noclip")
                .help("toggle collision in fly mode")
                .executes(|_| vec![UiEffect::ToggleNoclip]),
        );
        dispatcher.register(
            CommandNode::literal("save")
                .help("force an autosave now")
                .executes(|_| vec![UiEffect::Save, UiEffect::PostMessage("Saved.".into())]),
        );
        dispatcher.register(
            CommandNode::literal("help")
                .help("list commands")
                .executes({
                    let help_lines = Arc::clone(&help_lines);
                    move |_| {
                        help_lines
                            .get()
                            .map(|lines| lines.iter().cloned().map(UiEffect::PostMessage).collect())
                            .unwrap_or_else(|| {
                                vec![UiEffect::PostMessage("help is not available".into())]
                            })
                    }
                }),
        );
        dispatcher.register(
            CommandNode::literal("clear")
                .help("clear the chat log")
                .executes(|_| vec![UiEffect::ClearChat]),
        );

        let mut lines = vec!["Commands:".to_string()];
        lines.extend(dispatcher.help_lines(&source));
        let _ = help_lines.set(lines);

        Self { dispatcher, source }
    }

    pub fn dispatch(&self, line: &str) -> Result<Vec<UiEffect>, CommandError> {
        let body = command_body(line);
        self.dispatcher.execute(body, &self.source)
    }

    pub fn complete(&self, line: &str, cursor: usize) -> CompletionList {
        let (body, body_cursor, offset) = command_body_and_cursor(line, cursor);
        let mut completions = self.dispatcher.complete(body, body_cursor, &self.source);
        completions.range.start += offset;
        completions.range.end += offset;
        completions
    }

    pub fn hint(&self, line: &str, cursor: usize) -> Option<CommandHint> {
        if !line.trim_start().starts_with('/') {
            return None;
        }
        let (body, body_cursor, offset) = command_body_and_cursor(line, cursor);
        self.dispatcher
            .hint(body, body_cursor, &self.source)
            .map(|mut hint| {
                hint.cursor += offset;
                hint
            })
    }
}

fn command_body(line: &str) -> &str {
    let trimmed = line.trim();
    trimmed.strip_prefix('/').unwrap_or(trimmed)
}

fn command_body_and_cursor(line: &str, cursor: usize) -> (&str, usize, usize) {
    let cursor = cursor.min(line.len());
    if let Some(stripped) = line.strip_prefix('/') {
        (stripped, cursor.saturating_sub(1), 1)
    } else {
        (line, cursor, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_known_command() {
        let r = Registry::builtin();
        let effs = r.dispatch("/save").unwrap();
        assert!(effs.iter().any(|e| matches!(e, UiEffect::Save)));
    }

    #[test]
    fn dispatch_unknown_returns_error() {
        let r = Registry::builtin();
        let err = r.dispatch("/nope").unwrap_err();
        assert!(err.message.contains("unknown"));
    }

    #[test]
    fn dispatch_strips_leading_slash_optional() {
        let r = Registry::builtin();
        let effs = r.dispatch("tp 1 2 3").unwrap();
        assert_eq!(
            effs,
            vec![UiEffect::Teleport(glam::Vec3::new(1.0, 2.0, 3.0))]
        );
    }

    #[test]
    fn cmd_tp_parses_three_floats() {
        let r = Registry::builtin();
        let effs = r.dispatch("/tp 1.5 64 -32").unwrap();
        assert_eq!(
            effs,
            vec![UiEffect::Teleport(glam::Vec3::new(1.5, 64.0, -32.0))]
        );
    }

    #[test]
    fn cmd_tp_rejects_bad_args() {
        let r = Registry::builtin();
        assert!(r.dispatch("/tp 1 2 abc").is_err());
        let err = r.dispatch("/tp 1 2").unwrap_err();
        assert_eq!(err.kind, crate::command::CommandErrorKind::Expected);
        assert!(!err.message.contains("unknown"));
    }

    #[test]
    fn incomplete_tp_commands_are_expected_not_unknown() {
        let r = Registry::builtin();
        for line in ["/tp", "/teleport"] {
            let err = r.dispatch(line).unwrap_err();
            assert_eq!(err.kind, crate::command::CommandErrorKind::Expected);
            assert!(
                !err.message.contains("unknown"),
                "{line} produced {:?}",
                err.message
            );
        }
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
    fn zero_arg_commands_reject_extra_args() {
        let r = Registry::builtin();
        assert!(r.dispatch("/fly now").is_err());
        assert!(r.dispatch("/save now").is_err());
        assert!(r.dispatch("/clear now").is_err());
    }

    #[test]
    fn teleport_alias_redirects_to_tp() {
        let r = Registry::builtin();
        let effs = r.dispatch("/teleport 1 2 3").unwrap();
        assert_eq!(
            effs,
            vec![UiEffect::Teleport(glam::Vec3::new(1.0, 2.0, 3.0))]
        );
    }

    #[test]
    fn help_uses_generated_usage() {
        let r = Registry::builtin();
        let effs = r.dispatch("/help").unwrap();
        let lines: Vec<_> = effs
            .into_iter()
            .filter_map(|eff| match eff {
                UiEffect::PostMessage(msg) => Some(msg),
                _ => None,
            })
            .collect();
        assert!(
            lines
                .iter()
                .any(|line| line == "/tp <x> <y> <z> - teleport the player")
        );
        assert_eq!(lines.first().map(String::as_str), Some("Commands:"));
        assert!(lines.iter().any(|line| line == "/teleport - alias for /tp"));
    }

    #[test]
    fn completion_adjusts_for_slash() {
        let r = Registry::builtin();
        let completions = r.complete("/t", 2);
        assert_eq!(completions.range, 1..2);
        assert!(
            completions
                .entries
                .iter()
                .any(|entry| entry.replacement == "time")
        );
    }

    #[test]
    fn live_hint_reports_cursor_aware_error() {
        let r = Registry::builtin();
        let hint = r.hint("/tp 1 nope", 10).unwrap();
        assert!(hint.is_error);
        assert_eq!(hint.cursor, 6);
    }
}
