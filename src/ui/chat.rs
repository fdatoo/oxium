//! Chat log + single-line input field. The log is a fixed-capacity ring
//! buffer; the input is a byte-cursor-with-history text editor.

use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineKind {
    Player,
    System,
    CommandEcho,
    CommandError,
}

#[derive(Debug, Clone)]
pub struct ChatLine {
    pub text: String,
    pub posted_at: Instant,
    pub kind: LineKind,
}

const LOG_CAP: usize = 64;

#[derive(Debug, Default)]
pub struct ChatLog {
    pub(crate) lines: VecDeque<ChatLine>,
}

impl ChatLog {
    pub fn new() -> Self {
        Self { lines: VecDeque::with_capacity(LOG_CAP) }
    }
    pub fn push(&mut self, kind: LineKind, text: impl Into<String>) {
        if self.lines.len() == LOG_CAP {
            self.lines.pop_front();
        }
        self.lines.push_back(ChatLine {
            text: text.into(),
            posted_at: Instant::now(),
            kind,
        });
    }

    pub fn push_player(&mut self, text: impl Into<String>) {
        self.push(LineKind::Player, text);
    }

    pub fn push_system(&mut self, text: impl Into<String>) {
        self.push(LineKind::System, text);
    }

    pub fn push_echo(&mut self, text: impl Into<String>) {
        self.push(LineKind::CommandEcho, text);
    }

    pub fn push_error(&mut self, text: impl Into<String>) {
        self.push(LineKind::CommandError, text);
    }

    pub fn clear(&mut self) {
        self.lines.clear();
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &ChatLine> {
        self.lines.iter()
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_appends_and_classifies() {
        let mut log = ChatLog::new();
        log.push_player("hello");
        log.push_system("greetings");
        log.push_error("nope");
        assert_eq!(log.len(), 3);
        let lines: Vec<_> = log.iter().collect();
        assert_eq!(lines[0].kind, LineKind::Player);
        assert_eq!(lines[1].kind, LineKind::System);
        assert_eq!(lines[2].kind, LineKind::CommandError);
    }

    #[test]
    fn ring_evicts_oldest_at_capacity() {
        let mut log = ChatLog::new();
        for i in 0..LOG_CAP + 5 {
            log.push_player(format!("line {i}"));
        }
        assert_eq!(log.len(), LOG_CAP);
        let first = log.iter().next().unwrap();
        assert_eq!(first.text, format!("line {}", 5));
    }

    #[test]
    fn clear_empties_log() {
        let mut log = ChatLog::new();
        log.push_player("hi");
        log.clear();
        assert!(log.is_empty());
    }
}

#[derive(Debug, Default)]
pub struct ChatInput {
    pub buf: String,
    pub cursor: usize,
    pub history: VecDeque<String>,
    pub history_pos: Option<usize>,
}

impl ChatInput {
    pub fn new(prefill: &str) -> Self {
        Self { buf: prefill.to_string(), cursor: prefill.len(), ..Self::default() }
    }
}
