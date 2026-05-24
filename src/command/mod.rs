//! Rust-native command tree parser.
//!
//! The module is intentionally independent of UI and game state. Callers
//! provide a source type and command output type, then register literal and
//! argument nodes that produce that output.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

type Executor<S, O> = dyn Fn(&CommandContext<'_, S>) -> O + Send + Sync + 'static;
type Requirement<S> = dyn Fn(&S) -> bool + Send + Sync + 'static;

#[derive(Debug, Clone, PartialEq)]
pub enum ArgValue {
    F32(f32),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedArg {
    pub value: ArgValue,
    pub range: Range<usize>,
}

#[derive(Debug)]
pub struct CommandContext<'a, S> {
    source: &'a S,
    args: HashMap<String, ParsedArg>,
}

impl<'a, S> CommandContext<'a, S> {
    fn new(source: &'a S, args: HashMap<String, ParsedArg>) -> Self {
        Self { source, args }
    }

    pub fn source(&self) -> &'a S {
        self.source
    }

    pub fn arg(&self, name: &str) -> Option<&ParsedArg> {
        self.args.get(name)
    }

    pub fn f32(&self, name: &str) -> Option<f32> {
        match self.args.get(name).map(|arg| &arg.value) {
            Some(ArgValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn bool(&self, name: &str) -> Option<bool> {
        match self.args.get(name).map(|arg| &arg.value) {
            Some(ArgValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn string(&self, name: &str) -> Option<&str> {
        match self.args.get(name).map(|arg| &arg.value) {
            Some(ArgValue::String(v)) => Some(v.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandErrorKind {
    Empty,
    UnknownCommand,
    Expected,
    InvalidArgument,
    PermissionDenied,
    UnexpectedInput,
    NoExecutor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandError {
    pub kind: CommandErrorKind,
    pub message: String,
    pub cursor: usize,
    pub expected: Vec<String>,
}

impl CommandError {
    fn new(kind: CommandErrorKind, message: impl Into<String>, cursor: usize) -> Self {
        Self {
            kind,
            message: message.into(),
            cursor,
            expected: Vec::new(),
        }
    }

    fn expected(cursor: usize, expected: Vec<String>) -> Self {
        let message = if expected.is_empty() {
            "expected command".to_string()
        } else {
            format!("expected {}", expected.join(" or "))
        };
        Self {
            kind: CommandErrorKind::Expected,
            message,
            cursor,
            expected,
        }
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CommandError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionKind {
    Literal,
    Argument,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub replacement: String,
    pub display: String,
    pub hint: Option<String>,
    pub kind: CompletionKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionList {
    pub range: Range<usize>,
    pub entries: Vec<Completion>,
}

impl CompletionList {
    pub fn empty(cursor: usize) -> Self {
        Self {
            range: cursor..cursor,
            entries: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandHint {
    pub message: String,
    pub cursor: usize,
    pub is_error: bool,
}

pub trait ArgumentType: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn parse(&self, input: &str, cursor: usize) -> Result<(ArgValue, usize), CommandError>;
    fn examples(&self) -> &'static [&'static str] {
        &[]
    }
    fn complete(&self, input: &str, range: Range<usize>) -> Vec<Completion> {
        let prefix = &input[range.clone()];
        self.examples()
            .iter()
            .filter(|example| example.starts_with(prefix))
            .map(|example| Completion {
                replacement: (*example).to_string(),
                display: (*example).to_string(),
                hint: Some(self.name().to_string()),
                kind: CompletionKind::Argument,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct F32Argument;

impl ArgumentType for F32Argument {
    fn name(&self) -> &'static str {
        "number"
    }

    fn parse(&self, input: &str, cursor: usize) -> Result<(ArgValue, usize), CommandError> {
        let end = token_end(input, cursor);
        if end == cursor {
            return Err(CommandError::new(
                CommandErrorKind::Expected,
                "expected number",
                cursor,
            ));
        }
        let token = &input[cursor..end];
        let value = token.parse::<f32>().map_err(|_| {
            CommandError::new(
                CommandErrorKind::InvalidArgument,
                format!("not a number: {token}"),
                cursor,
            )
        })?;
        if !value.is_finite() {
            return Err(CommandError::new(
                CommandErrorKind::InvalidArgument,
                format!("not a finite number: {token}"),
                cursor,
            ));
        }
        Ok((ArgValue::F32(value), end))
    }

    fn examples(&self) -> &'static [&'static str] {
        &["0", "1", "64", "-32"]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct F32RangeArgument {
    min: f32,
    max: f32,
}

impl F32RangeArgument {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

impl ArgumentType for F32RangeArgument {
    fn name(&self) -> &'static str {
        "number"
    }

    fn parse(&self, input: &str, cursor: usize) -> Result<(ArgValue, usize), CommandError> {
        let (value, end) = F32Argument.parse(input, cursor)?;
        let ArgValue::F32(value) = value else {
            unreachable!("F32Argument only returns f32");
        };
        if value < self.min || value > self.max {
            return Err(CommandError::new(
                CommandErrorKind::InvalidArgument,
                format!("expected {}..={}, got {value}", self.min, self.max),
                cursor,
            ));
        }
        Ok((ArgValue::F32(value), end))
    }

    fn examples(&self) -> &'static [&'static str] {
        &["0", "0.25", "0.5", "1"]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoolArgument;

impl ArgumentType for BoolArgument {
    fn name(&self) -> &'static str {
        "bool"
    }

    fn parse(&self, input: &str, cursor: usize) -> Result<(ArgValue, usize), CommandError> {
        let end = token_end(input, cursor);
        let token = &input[cursor..end];
        let value = match token {
            "true" | "on" => true,
            "false" | "off" => false,
            _ => {
                return Err(CommandError::new(
                    CommandErrorKind::InvalidArgument,
                    format!("expected true or false, got {token}"),
                    cursor,
                ));
            }
        };
        Ok((ArgValue::Bool(value), end))
    }

    fn examples(&self) -> &'static [&'static str] {
        &["true", "false", "on", "off"]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WordArgument;

impl ArgumentType for WordArgument {
    fn name(&self) -> &'static str {
        "word"
    }

    fn parse(&self, input: &str, cursor: usize) -> Result<(ArgValue, usize), CommandError> {
        let end = token_end(input, cursor);
        if end == cursor {
            return Err(CommandError::new(
                CommandErrorKind::Expected,
                "expected word",
                cursor,
            ));
        }
        Ok((ArgValue::String(input[cursor..end].to_string()), end))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RestStringArgument;

impl ArgumentType for RestStringArgument {
    fn name(&self) -> &'static str {
        "text"
    }

    fn parse(&self, input: &str, cursor: usize) -> Result<(ArgValue, usize), CommandError> {
        if cursor >= input.len() {
            return Err(CommandError::new(
                CommandErrorKind::Expected,
                "expected text",
                cursor,
            ));
        }
        Ok((ArgValue::String(input[cursor..].to_string()), input.len()))
    }
}

enum NodeKind {
    Root,
    Literal(String),
    Argument {
        name: String,
        parser: Box<dyn ArgumentType>,
    },
}

impl NodeKind {
    fn display(&self) -> String {
        match self {
            Self::Root => String::new(),
            Self::Literal(name) => name.clone(),
            Self::Argument { name, .. } => format!("<{name}>"),
        }
    }

    fn suggestion_label(&self) -> String {
        match self {
            Self::Root => String::new(),
            Self::Literal(name) => name.clone(),
            Self::Argument { name, parser } => format!("<{name}: {}>", parser.name()),
        }
    }
}

pub struct CommandNode<S, O> {
    kind: NodeKind,
    children: Vec<CommandNode<S, O>>,
    executor: Option<Arc<Executor<S, O>>>,
    requirement: Arc<Requirement<S>>,
    redirect: Option<Vec<String>>,
    help: Option<String>,
}

impl<S, O> CommandNode<S, O> {
    fn root() -> Self {
        Self {
            kind: NodeKind::Root,
            children: Vec::new(),
            executor: None,
            requirement: Arc::new(|_| true),
            redirect: None,
            help: None,
        }
    }

    pub fn literal(name: impl Into<String>) -> Self {
        Self {
            kind: NodeKind::Literal(name.into()),
            children: Vec::new(),
            executor: None,
            requirement: Arc::new(|_| true),
            redirect: None,
            help: None,
        }
    }

    pub fn argument(name: impl Into<String>, parser: impl ArgumentType) -> Self {
        Self {
            kind: NodeKind::Argument {
                name: name.into(),
                parser: Box::new(parser),
            },
            children: Vec::new(),
            executor: None,
            requirement: Arc::new(|_| true),
            redirect: None,
            help: None,
        }
    }

    pub fn then(mut self, child: CommandNode<S, O>) -> Self {
        self.children.push(child);
        self
    }

    pub fn executes(
        mut self,
        executor: impl Fn(&CommandContext<'_, S>) -> O + Send + Sync + 'static,
    ) -> Self {
        self.executor = Some(Arc::new(executor));
        self
    }

    pub fn requires(mut self, requirement: impl Fn(&S) -> bool + Send + Sync + 'static) -> Self {
        self.requirement = Arc::new(requirement);
        self
    }

    pub fn redirects_to(mut self, target: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.redirect = Some(target.into_iter().map(Into::into).collect());
        self
    }

    pub fn help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

pub struct ParseResult<'a, S, O> {
    context: CommandContext<'a, S>,
    executor: Arc<Executor<S, O>>,
    pub consumed_nodes: Vec<String>,
    pub redirects: Vec<Vec<String>>,
}

struct ParseScratch {
    args: HashMap<String, ParsedArg>,
    consumed_nodes: Vec<String>,
    redirects: Vec<Vec<String>>,
}

pub struct CommandDispatcher<S, O> {
    root: CommandNode<S, O>,
}

impl<S, O> Default for CommandDispatcher<S, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, O> CommandDispatcher<S, O> {
    pub fn new() -> Self {
        Self {
            root: CommandNode::root(),
        }
    }

    pub fn register(&mut self, node: CommandNode<S, O>) {
        self.root.children.push(node);
    }

    pub fn parse<'a>(
        &'a self,
        input: &'a str,
        source: &'a S,
    ) -> Result<ParseResult<'a, S, O>, CommandError> {
        let mut scratch = ParseScratch {
            args: HashMap::new(),
            consumed_nodes: Vec::new(),
            redirects: Vec::new(),
        };
        let cursor = skip_ws(input, 0);
        if cursor >= input.len() {
            return Err(CommandError::new(
                CommandErrorKind::Empty,
                "empty command",
                cursor,
            ));
        }
        self.parse_from_node(&self.root, input, cursor, source, &mut scratch)
            .map(|executor| ParseResult {
                context: CommandContext::new(source, scratch.args),
                executor,
                consumed_nodes: scratch.consumed_nodes,
                redirects: scratch.redirects,
            })
            .map_err(|err| {
                if matches!(err.kind, CommandErrorKind::Expected) && err.cursor == cursor {
                    CommandError {
                        kind: CommandErrorKind::UnknownCommand,
                        message: unknown_message(input, cursor),
                        cursor,
                        expected: err.expected,
                    }
                } else {
                    err
                }
            })
    }

    pub fn execute(&self, input: &str, source: &S) -> Result<O, CommandError> {
        let parsed = self.parse(input, source)?;
        Ok((parsed.executor)(&parsed.context))
    }

    pub fn complete(&self, input: &str, cursor: usize, source: &S) -> CompletionList {
        let cursor = cursor.min(input.len());
        let mut node = &self.root;
        let mut pos = skip_ws(input, 0);

        while pos < cursor {
            let token_start = pos;
            let token_end = token_end(input, token_start).min(cursor);
            let partial = token_end == cursor && !ends_with_ws(input, cursor);
            if partial {
                return self.complete_children(node, input, token_start..cursor, source);
            }

            let Some((child, end)) = self.match_child(node, input, token_start, source) else {
                return self.complete_children(node, input, token_start..cursor, source);
            };
            node = self.redirect_target(child).unwrap_or(child);
            pos = skip_ws(input, end);
        }

        let range = if cursor > 0 && !ends_with_ws(input, cursor) {
            token_start_before(input, cursor)..cursor
        } else {
            cursor..cursor
        };
        self.complete_children(node, input, range, source)
    }

    pub fn hint(&self, input: &str, cursor: usize, source: &S) -> Option<CommandHint> {
        if input.trim().is_empty() {
            return Some(CommandHint {
                message: "type a command".to_string(),
                cursor,
                is_error: false,
            });
        }
        match self.parse(input, source) {
            Ok(parsed) => Some(CommandHint {
                message: format!("ready: /{}", parsed.consumed_nodes.join(" ")),
                cursor,
                is_error: false,
            }),
            Err(err) => Some(CommandHint {
                message: err.message,
                cursor: err.cursor,
                is_error: true,
            }),
        }
    }

    pub fn usages(&self, source: &S) -> Vec<String> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.collect_usages(&self.root, source, &mut path, &mut out);
        out
    }

    pub fn help_lines(&self, source: &S) -> Vec<String> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.collect_help(&self.root, source, &mut path, &mut out);
        out
    }

    pub fn find_ambiguities(&self, source: &S) -> Vec<String> {
        let mut out = Vec::new();
        let mut path = Vec::new();
        self.find_ambiguities_at(&self.root, source, &mut path, &mut out);
        out
    }

    fn parse_from_node(
        &self,
        node: &CommandNode<S, O>,
        input: &str,
        cursor: usize,
        source: &S,
        scratch: &mut ParseScratch,
    ) -> Result<Arc<Executor<S, O>>, CommandError> {
        if let Some(target) = &node.redirect {
            let Some(target_node) = self.find_path(target) else {
                return Err(CommandError::new(
                    CommandErrorKind::NoExecutor,
                    format!("redirect target not found: {}", target.join(" ")),
                    cursor,
                ));
            };
            scratch.redirects.push(target.clone());
            return self.parse_from_node(target_node, input, cursor, source, scratch);
        }

        let cursor = skip_ws(input, cursor);
        if cursor >= input.len() {
            if let Some(executor) = &node.executor {
                return Ok(executor.clone());
            }
            return Err(CommandError::expected(
                cursor,
                self.expected_children(node, source),
            ));
        }

        let mut best_error: Option<CommandError> = None;
        for child in node
            .children
            .iter()
            .filter(|child| (child.requirement)(source))
        {
            let checkpoint_len = scratch.consumed_nodes.len();
            let checkpoint_args = scratch.args.clone();
            match self.consume_child(child, input, cursor, &mut scratch.args) {
                Ok(end) => {
                    scratch.consumed_nodes.push(child.kind.display());
                    match self.parse_from_node(child, input, end, source, scratch) {
                        Ok(executor) => return Ok(executor),
                        Err(err) => best_error = Some(better_error(best_error, err)),
                    }
                }
                Err(err) => best_error = Some(better_error(best_error, err)),
            }
            scratch.consumed_nodes.truncate(checkpoint_len);
            scratch.args = checkpoint_args;
        }

        if node.executor.is_some() {
            return Err(CommandError::new(
                CommandErrorKind::UnexpectedInput,
                format!(
                    "unexpected input: {}",
                    &input[cursor..token_end(input, cursor)]
                ),
                cursor,
            ));
        }

        match best_error {
            Some(err) if err.cursor > cursor || !matches!(err.kind, CommandErrorKind::Expected) => {
                Err(err)
            }
            _ => Err(CommandError::expected(
                cursor,
                self.expected_children(node, source),
            )),
        }
    }

    fn consume_child(
        &self,
        child: &CommandNode<S, O>,
        input: &str,
        cursor: usize,
        args: &mut HashMap<String, ParsedArg>,
    ) -> Result<usize, CommandError> {
        match &child.kind {
            NodeKind::Root => Ok(cursor),
            NodeKind::Literal(name) => {
                let end = token_end(input, cursor);
                let token = &input[cursor..end];
                if token == name {
                    Ok(end)
                } else {
                    Err(CommandError::new(
                        CommandErrorKind::Expected,
                        format!("expected {name}"),
                        cursor,
                    ))
                }
            }
            NodeKind::Argument { name, parser } => {
                let start = cursor;
                let (value, end) = parser.parse(input, cursor)?;
                args.insert(
                    name.clone(),
                    ParsedArg {
                        value,
                        range: start..end,
                    },
                );
                Ok(end)
            }
        }
    }

    fn match_child<'a>(
        &'a self,
        node: &'a CommandNode<S, O>,
        input: &str,
        cursor: usize,
        source: &S,
    ) -> Option<(&'a CommandNode<S, O>, usize)> {
        for child in node
            .children
            .iter()
            .filter(|child| (child.requirement)(source))
        {
            let mut args = HashMap::new();
            if let Ok(end) = self.consume_child(child, input, cursor, &mut args) {
                return Some((child, end));
            }
        }
        None
    }

    fn complete_children(
        &self,
        node: &CommandNode<S, O>,
        input: &str,
        range: Range<usize>,
        source: &S,
    ) -> CompletionList {
        let prefix = &input[range.clone()];
        let mut entries = Vec::new();
        for child in node
            .children
            .iter()
            .filter(|child| (child.requirement)(source))
        {
            match &child.kind {
                NodeKind::Literal(name) if name.starts_with(prefix) => entries.push(Completion {
                    replacement: name.clone(),
                    display: name.clone(),
                    hint: child.help.clone(),
                    kind: CompletionKind::Literal,
                }),
                NodeKind::Argument { parser, .. } => {
                    entries.extend(parser.complete(input, range.clone()))
                }
                _ => {}
            }
        }
        entries.sort_by(|a, b| a.display.cmp(&b.display));
        CompletionList { range, entries }
    }

    fn redirect_target<'a>(&'a self, node: &'a CommandNode<S, O>) -> Option<&'a CommandNode<S, O>> {
        node.redirect.as_ref().and_then(|path| self.find_path(path))
    }

    fn find_path(&self, path: &[String]) -> Option<&CommandNode<S, O>> {
        let mut node = &self.root;
        for segment in path {
            node = node.children.iter().find(|child| match &child.kind {
                NodeKind::Literal(name) => name == segment,
                _ => false,
            })?;
        }
        Some(node)
    }

    fn expected_children(&self, node: &CommandNode<S, O>, source: &S) -> Vec<String> {
        node.children
            .iter()
            .filter(|child| (child.requirement)(source))
            .map(|child| child.kind.suggestion_label())
            .collect()
    }

    fn collect_usages(
        &self,
        node: &CommandNode<S, O>,
        source: &S,
        path: &mut Vec<String>,
        out: &mut Vec<String>,
    ) {
        if !(node.requirement)(source) {
            return;
        }
        if (node.executor.is_some() || node.redirect.is_some()) && !path.is_empty() {
            out.push(path.join(" "));
        }
        for child in &node.children {
            path.push(child.kind.display());
            self.collect_usages(child, source, path, out);
            path.pop();
        }
    }

    fn collect_help(
        &self,
        node: &CommandNode<S, O>,
        source: &S,
        path: &mut Vec<String>,
        out: &mut Vec<String>,
    ) {
        if !(node.requirement)(source) {
            return;
        }
        if (node.executor.is_some() || node.redirect.is_some()) && !path.is_empty() {
            let usage = path.join(" ");
            if let Some(help) = &node.help {
                out.push(format!("/{usage} - {help}"));
            } else {
                out.push(format!("/{usage}"));
            }
        }
        for child in &node.children {
            path.push(child.kind.display());
            self.collect_help(child, source, path, out);
            path.pop();
        }
    }

    fn find_ambiguities_at(
        &self,
        node: &CommandNode<S, O>,
        source: &S,
        path: &mut Vec<String>,
        out: &mut Vec<String>,
    ) {
        if !(node.requirement)(source) {
            return;
        }
        for (i, left) in node.children.iter().enumerate() {
            for right in node.children.iter().skip(i + 1) {
                if self.nodes_overlap(left, right) {
                    let here = if path.is_empty() {
                        "<root>".to_string()
                    } else {
                        path.join(" ")
                    };
                    out.push(format!(
                        "{here}: {} overlaps {}",
                        left.kind.suggestion_label(),
                        right.kind.suggestion_label()
                    ));
                }
            }
        }
        for child in &node.children {
            path.push(child.kind.display());
            self.find_ambiguities_at(child, source, path, out);
            path.pop();
        }
    }

    fn nodes_overlap(&self, left: &CommandNode<S, O>, right: &CommandNode<S, O>) -> bool {
        match (&left.kind, &right.kind) {
            (NodeKind::Literal(a), NodeKind::Literal(b)) => a == b,
            (NodeKind::Literal(lit), NodeKind::Argument { parser, .. })
            | (NodeKind::Argument { parser, .. }, NodeKind::Literal(lit)) => {
                parser.parse(lit, 0).is_ok()
            }
            (NodeKind::Argument { parser: a, .. }, NodeKind::Argument { parser: b, .. }) => a
                .examples()
                .iter()
                .chain(b.examples().iter())
                .any(|example| a.parse(example, 0).is_ok() && b.parse(example, 0).is_ok()),
            _ => false,
        }
    }
}

fn better_error(current: Option<CommandError>, next: CommandError) -> CommandError {
    match current {
        Some(current) if current.cursor > next.cursor => current,
        Some(current)
            if current.cursor == next.cursor
                && !matches!(next.kind, CommandErrorKind::InvalidArgument) =>
        {
            current
        }
        _ => next,
    }
}

fn skip_ws(input: &str, mut cursor: usize) -> usize {
    while cursor < input.len() {
        let ch = input[cursor..].chars().next().unwrap();
        if !ch.is_whitespace() {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn token_end(input: &str, mut cursor: usize) -> usize {
    while cursor < input.len() {
        let ch = input[cursor..].chars().next().unwrap();
        if ch.is_whitespace() {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn token_start_before(input: &str, cursor: usize) -> usize {
    let mut start = 0;
    for (idx, ch) in input[..cursor].char_indices() {
        if ch.is_whitespace() {
            start = idx + ch.len_utf8();
        }
    }
    start
}

fn ends_with_ws(input: &str, cursor: usize) -> bool {
    input[..cursor]
        .chars()
        .next_back()
        .is_some_and(char::is_whitespace)
}

fn unknown_message(input: &str, cursor: usize) -> String {
    let end = token_end(input, cursor);
    format!("unknown command: /{}", &input[cursor..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Source {
        admin: bool,
    }

    fn dispatcher() -> CommandDispatcher<Source, String> {
        let mut d = CommandDispatcher::new();
        d.register(
            CommandNode::literal("echo").then(
                CommandNode::argument("text", RestStringArgument)
                    .help("echo text")
                    .executes(|ctx| ctx.string("text").unwrap().to_string()),
            ),
        );
        d.register(
            CommandNode::literal("tp").then(
                CommandNode::argument("x", F32Argument).then(
                    CommandNode::argument("y", F32Argument).then(
                        CommandNode::argument("z", F32Argument)
                            .help("teleport")
                            .executes(|ctx| {
                                format!(
                                    "{},{},{}",
                                    ctx.f32("x").unwrap(),
                                    ctx.f32("y").unwrap(),
                                    ctx.f32("z").unwrap()
                                )
                            }),
                    ),
                ),
            ),
        );
        d.register(
            CommandNode::literal("teleport")
                .redirects_to(["tp"])
                .help("alias for /tp"),
        );
        d.register(
            CommandNode::literal("admin")
                .requires(|source: &Source| source.admin)
                .help("admin only")
                .executes(|_| "admin".to_string()),
        );
        d
    }

    #[test]
    fn parses_typed_arguments() {
        let d = dispatcher();
        let out = d.execute("tp 1 2 -3", &Source::default()).unwrap();
        assert_eq!(out, "1,2,-3");
    }

    #[test]
    fn redirects_alias_to_target_children() {
        let d = dispatcher();
        let out = d.execute("teleport 1 2 3", &Source::default()).unwrap();
        assert_eq!(out, "1,2,3");
    }

    #[test]
    fn incomplete_literal_reports_expected_child_not_unknown() {
        let d = dispatcher();
        let err = d.execute("tp", &Source::default()).unwrap_err();
        assert_eq!(err.kind, CommandErrorKind::Expected);
        assert_eq!(err.cursor, 2);
        assert!(!err.message.contains("unknown"));
    }

    #[test]
    fn reports_bad_number_cursor() {
        let d = dispatcher();
        let err = d.execute("tp 1 nope 3", &Source::default()).unwrap_err();
        assert_eq!(err.kind, CommandErrorKind::InvalidArgument);
        assert_eq!(err.cursor, 5);
    }

    #[test]
    fn reports_trailing_input() {
        let d = dispatcher();
        let err = d.execute("tp 1 2 3 extra", &Source::default()).unwrap_err();
        assert_eq!(err.kind, CommandErrorKind::UnexpectedInput);
        assert_eq!(err.cursor, 9);
    }

    #[test]
    fn completion_lists_literals() {
        let d = dispatcher();
        let list = d.complete("t", 1, &Source::default());
        assert_eq!(list.range, 0..1);
        assert!(list.entries.iter().any(|entry| entry.replacement == "tp"));
        assert!(
            list.entries
                .iter()
                .any(|entry| entry.replacement == "teleport")
        );
    }

    #[test]
    fn source_predicate_filters_execution_and_completion() {
        let d = dispatcher();
        assert!(d.execute("admin", &Source::default()).is_err());
        let list = d.complete("a", 1, &Source::default());
        assert!(list.entries.is_empty());
        let list = d.complete("a", 1, &Source { admin: true });
        assert_eq!(list.entries[0].replacement, "admin");
    }

    #[test]
    fn help_is_generated_from_tree() {
        let d = dispatcher();
        let lines = d.help_lines(&Source::default());
        assert!(
            lines
                .iter()
                .any(|line| line == "/tp <x> <y> <z> - teleport")
        );
        assert!(lines.iter().any(|line| line == "/teleport - alias for /tp"));
    }

    #[test]
    fn ambiguity_detection_finds_literal_argument_overlap() {
        let mut d: CommandDispatcher<Source, ()> = CommandDispatcher::new();
        d.register(CommandNode::literal("root").then(CommandNode::literal("1").executes(|_| ())));
        d.register(
            CommandNode::literal("root")
                .then(CommandNode::argument("value", F32Argument).executes(|_| ())),
        );
        let ambiguities = d.find_ambiguities(&Source::default());
        assert!(!ambiguities.is_empty());
    }
}
