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

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    #[cfg(test)]
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

    #[test]
    fn insert_appends_at_cursor() {
        let mut ci = ChatInput::default();
        ci.insert_text("hello");
        assert_eq!(ci.buf, "hello");
        assert_eq!(ci.cursor, 5);
        ci.move_home();
        ci.insert_text("> ");
        assert_eq!(ci.buf, "> hello");
        assert_eq!(ci.cursor, 2);
    }

    #[test]
    fn backspace_handles_multibyte() {
        let mut ci = ChatInput::default();
        ci.insert_text("héllo");
        ci.backspace();
        assert_eq!(ci.buf, "héll");
        ci.move_home();
        ci.move_right();
        ci.backspace();
        assert_eq!(ci.buf, "éll");
    }

    #[test]
    fn delete_at_end_no_op() {
        let mut ci = ChatInput::default();
        ci.insert_text("ab");
        ci.delete_forward();
        assert_eq!(ci.buf, "ab");
    }

    #[test]
    fn insert_filters_control_chars() {
        let mut ci = ChatInput::default();
        ci.insert_text("a\nb\tc");
        assert_eq!(ci.buf, "abc");
    }

    #[test]
    fn submit_returns_and_records_history() {
        let mut ci = ChatInput::default();
        ci.insert_text("/help");
        let out = ci.submit();
        assert_eq!(out, "/help");
        assert_eq!(ci.buf, "");
        ci.history_prev();
        assert_eq!(ci.buf, "/help");
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

    pub fn insert_text(&mut self, text: &str) {
        for ch in text.chars() {
            if ch.is_control() { continue }
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            self.buf.insert_str(self.cursor, s);
            self.cursor += s.len();
        }
        self.history_pos = None;
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 { return }
        let mut new_cursor = self.cursor - 1;
        while !self.buf.is_char_boundary(new_cursor) && new_cursor > 0 {
            new_cursor -= 1;
        }
        self.buf.replace_range(new_cursor..self.cursor, "");
        self.cursor = new_cursor;
        self.history_pos = None;
    }

    pub fn delete_forward(&mut self) {
        if self.cursor >= self.buf.len() { return }
        let mut end = self.cursor + 1;
        while end < self.buf.len() && !self.buf.is_char_boundary(end) {
            end += 1;
        }
        self.buf.replace_range(self.cursor..end, "");
        self.history_pos = None;
    }

    pub fn move_left(&mut self) {
        if self.cursor == 0 { return }
        let mut c = self.cursor - 1;
        while c > 0 && !self.buf.is_char_boundary(c) { c -= 1; }
        self.cursor = c;
    }

    pub fn move_right(&mut self) {
        if self.cursor >= self.buf.len() { return }
        let mut c = self.cursor + 1;
        while c < self.buf.len() && !self.buf.is_char_boundary(c) { c += 1; }
        self.cursor = c;
    }

    pub fn move_home(&mut self) { self.cursor = 0; }
    pub fn move_end(&mut self)  { self.cursor = self.buf.len(); }

    pub fn history_prev(&mut self) {
        if self.history.is_empty() { return }
        let next = match self.history_pos {
            None       => self.history.len() - 1,
            Some(0)    => 0,
            Some(i)    => i - 1,
        };
        self.history_pos = Some(next);
        self.buf = self.history[next].clone();
        self.cursor = self.buf.len();
    }

    pub fn history_next(&mut self) {
        match self.history_pos {
            None => {}
            Some(i) if i + 1 >= self.history.len() => {
                self.history_pos = None;
                self.buf.clear();
                self.cursor = 0;
            }
            Some(i) => {
                self.history_pos = Some(i + 1);
                self.buf = self.history[i + 1].clone();
                self.cursor = self.buf.len();
            }
        }
    }

    pub fn submit(&mut self) -> String {
        let line = std::mem::take(&mut self.buf);
        self.cursor = 0;
        self.history_pos = None;
        if !line.is_empty() {
            if self.history.len() == 32 { self.history.pop_front(); }
            self.history.push_back(line.clone());
        }
        line
    }
}
