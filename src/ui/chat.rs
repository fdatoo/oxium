//! Chat log + single-line input field. The log is a fixed-capacity ring
//! buffer; the input is a byte-cursor-with-history text editor.

use std::collections::VecDeque;
use std::ops::Range;
use std::time::Instant;

use crate::command::{Completion, CompletionList};

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
        Self {
            lines: VecDeque::with_capacity(LOG_CAP),
        }
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

    #[test]
    fn selection_replacement_and_delete() {
        let mut ci = ChatInput::new("hello world");
        ci.set_cursor(5, false);
        ci.set_cursor(11, true);
        assert_eq!(ci.selected_range(), Some(5..11));
        ci.replace_selection(" rust");
        assert_eq!(ci.buf, "hello rust");
        assert_eq!(ci.cursor, 10);
        assert_eq!(ci.selected_range(), None);

        ci.set_cursor(0, false);
        ci.set_cursor(5, true);
        ci.delete_forward();
        assert_eq!(ci.buf, " rust");
        assert_eq!(ci.cursor, 0);
    }

    #[test]
    fn select_all_replaces_everything() {
        let mut ci = ChatInput::new("/help");
        ci.select_all();
        ci.insert_text("/time");
        assert_eq!(ci.buf, "/time");
        assert_eq!(ci.cursor, 5);
        assert_eq!(ci.selected_range(), None);
    }

    #[test]
    fn word_navigation_and_deletion_are_byte_safe() {
        let mut ci = ChatInput::new("/teleport héllo, world");
        ci.move_end();
        ci.delete_word_backward();
        assert_eq!(ci.buf, "/teleport héllo, ");
        ci.delete_word_backward();
        assert_eq!(ci.buf, "/teleport héllo");
        ci.delete_word_backward();
        assert_eq!(ci.buf, "/teleport ");
        ci.move_home();
        ci.delete_word_forward();
        assert_eq!(ci.buf, "teleport ");
        ci.move_word_right(false);
        assert_eq!(ci.cursor, "teleport".len());
    }

    #[test]
    fn shift_selection_by_char_and_word() {
        let mut ci = ChatInput::new("alpha beta");
        ci.move_home();
        ci.move_word_right(false);
        ci.move_right_ext(true);
        assert_eq!(ci.selected_range(), Some(5..6));
        ci.move_word_right(true);
        assert_eq!(ci.selected_range(), Some(5..10));
        ci.move_home_ext(false);
        assert_eq!(ci.selected_range(), None);
    }

    #[test]
    fn word_range_at_handles_punctuation_and_multibyte() {
        let ci = ChatInput::new("/tp héllo,world");
        assert_eq!(ci.word_range_at(0), 0..1);
        let h = ci.buf.find('h').unwrap();
        assert_eq!(&ci.buf[ci.word_range_at(h)], "héllo");
        let comma = ci.buf.find(',').unwrap();
        assert_eq!(&ci.buf[ci.word_range_at(comma)], ",");
    }
}

#[derive(Debug, Default)]
pub struct ChatInput {
    pub buf: String,
    pub cursor: usize,
    pub selection_anchor: Option<usize>,
    pub history: VecDeque<String>,
    pub history_pos: Option<usize>,
    pub completion: CompletionState,
}

#[derive(Debug, Default)]
pub struct CompletionState {
    pub range: Option<Range<usize>>,
    pub entries: Vec<Completion>,
    pub selected: usize,
}

impl ChatInput {
    pub fn new(prefill: &str) -> Self {
        Self {
            buf: prefill.to_string(),
            cursor: prefill.len(),
            ..Self::default()
        }
    }

    pub fn selected_range(&self) -> Option<Range<usize>> {
        let anchor = self.selection_anchor?;
        if anchor == self.cursor {
            return None;
        }
        Some(anchor.min(self.cursor)..anchor.max(self.cursor))
    }

    pub fn set_cursor(&mut self, byte: usize, extend_selection: bool) {
        let byte = self.clamp_to_boundary(byte);
        if extend_selection {
            self.selection_anchor.get_or_insert(self.cursor);
        } else {
            self.selection_anchor = None;
        }
        self.cursor = byte;
        self.clear_completion();
    }

    pub fn select_all(&mut self) {
        self.selection_anchor = Some(0);
        self.cursor = self.buf.len();
        self.clear_completion();
    }

    pub fn replace_selection(&mut self, text: &str) {
        let replacement = filtered_text(text);
        if let Some(range) = self.selected_range() {
            self.buf.replace_range(range.clone(), &replacement);
            self.cursor = range.start + replacement.len();
        } else {
            self.buf.insert_str(self.cursor, &replacement);
            self.cursor += replacement.len();
        }
        self.selection_anchor = None;
        self.history_pos = None;
        self.clear_completion();
    }

    pub fn insert_text(&mut self, text: &str) {
        self.replace_selection(text);
    }

    pub fn backspace(&mut self) {
        self.delete_backward();
    }

    pub fn delete_backward(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor == 0 {
            return;
        }
        let new_cursor = prev_boundary(&self.buf, self.cursor);
        self.buf.replace_range(new_cursor..self.cursor, "");
        self.cursor = new_cursor;
        self.history_pos = None;
        self.clear_completion();
    }

    pub fn delete_forward(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor >= self.buf.len() {
            return;
        }
        let end = next_boundary(&self.buf, self.cursor);
        self.buf.replace_range(self.cursor..end, "");
        self.history_pos = None;
        self.clear_completion();
    }

    pub fn move_left(&mut self) {
        self.move_left_ext(false);
    }

    pub fn move_left_ext(&mut self, extend_selection: bool) {
        if self.cursor == 0 {
            if !extend_selection {
                self.selection_anchor = None;
            }
            return;
        }
        self.set_cursor(prev_boundary(&self.buf, self.cursor), extend_selection);
    }

    pub fn move_right(&mut self) {
        self.move_right_ext(false);
    }

    pub fn move_right_ext(&mut self, extend_selection: bool) {
        if self.cursor >= self.buf.len() {
            if !extend_selection {
                self.selection_anchor = None;
            }
            return;
        }
        self.set_cursor(next_boundary(&self.buf, self.cursor), extend_selection);
    }

    pub fn move_home(&mut self) {
        self.move_home_ext(false);
    }
    pub fn move_end(&mut self) {
        self.move_end_ext(false);
    }

    pub fn move_home_ext(&mut self, extend_selection: bool) {
        self.set_cursor(0, extend_selection);
    }

    pub fn move_end_ext(&mut self, extend_selection: bool) {
        self.set_cursor(self.buf.len(), extend_selection);
    }

    pub fn move_word_left(&mut self, extend_selection: bool) {
        let target = word_left_boundary(&self.buf, self.cursor);
        self.set_cursor(target, extend_selection);
    }

    pub fn move_word_right(&mut self, extend_selection: bool) {
        let target = word_right_boundary(&self.buf, self.cursor);
        self.set_cursor(target, extend_selection);
    }

    pub fn delete_word_backward(&mut self) {
        if self.delete_selection() {
            return;
        }
        let start = word_left_boundary(&self.buf, self.cursor);
        if start == self.cursor {
            return;
        }
        self.buf.replace_range(start..self.cursor, "");
        self.cursor = start;
        self.history_pos = None;
        self.clear_completion();
    }

    pub fn delete_word_forward(&mut self) {
        if self.delete_selection() {
            return;
        }
        let end = word_right_boundary(&self.buf, self.cursor);
        if end == self.cursor {
            return;
        }
        self.buf.replace_range(self.cursor..end, "");
        self.history_pos = None;
        self.clear_completion();
    }

    pub fn word_range_at(&self, byte: usize) -> Range<usize> {
        if self.buf.is_empty() {
            return 0..0;
        }
        let byte = self.clamp_to_boundary(byte);
        let pos = if byte < self.buf.len() {
            byte
        } else {
            prev_boundary(&self.buf, byte)
        };
        let Some(ch) = self.buf[pos..].chars().next() else {
            return byte..byte;
        };
        let class = word_class(ch);
        if class == WordClass::Space {
            return pos..pos;
        }

        let mut start = pos;
        while start > 0 {
            let prev = prev_boundary(&self.buf, start);
            let Some(ch) = self.buf[prev..start].chars().next() else {
                break;
            };
            if word_class(ch) != class {
                break;
            }
            start = prev;
        }

        let mut end = next_boundary(&self.buf, pos);
        while end < self.buf.len() {
            let next = next_boundary(&self.buf, end);
            let Some(ch) = self.buf[end..next].chars().next() else {
                break;
            };
            if word_class(ch) != class {
                break;
            }
            end = next;
        }
        start..end
    }

    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_pos {
            None => self.history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_pos = Some(next);
        self.buf = self.history[next].clone();
        self.cursor = self.buf.len();
        self.selection_anchor = None;
        self.clear_completion();
    }

    pub fn history_next(&mut self) {
        match self.history_pos {
            None => {}
            Some(i) if i + 1 >= self.history.len() => {
                self.history_pos = None;
                self.buf.clear();
                self.cursor = 0;
                self.selection_anchor = None;
                self.clear_completion();
            }
            Some(i) => {
                self.history_pos = Some(i + 1);
                self.buf = self.history[i + 1].clone();
                self.cursor = self.buf.len();
                self.selection_anchor = None;
                self.clear_completion();
            }
        }
    }

    pub fn apply_completion(&mut self, completions: CompletionList) -> bool {
        if completions.entries.is_empty() {
            self.clear_completion();
            return false;
        }

        let initial_range = self
            .selected_range()
            .unwrap_or_else(|| completions.range.clone());
        let (range, selected) = match &self.completion.range {
            Some(range) if self.completion.entries == completions.entries => (
                range.clone(),
                (self.completion.selected + 1) % completions.entries.len(),
            ),
            _ => (initial_range, 0),
        };

        let replacement = &completions.entries[selected].replacement;
        self.buf.replace_range(range.clone(), replacement);
        let end = range.start + replacement.len();
        self.cursor = end;
        self.selection_anchor = None;
        self.history_pos = None;
        self.completion = CompletionState {
            range: Some(range.start..end),
            entries: completions.entries,
            selected,
        };
        true
    }

    pub fn clear_completion(&mut self) {
        self.completion = CompletionState::default();
    }

    pub fn submit(&mut self) -> String {
        let line = std::mem::take(&mut self.buf);
        self.cursor = 0;
        self.selection_anchor = None;
        self.history_pos = None;
        self.clear_completion();
        if !line.is_empty() {
            if self.history.len() == 32 {
                self.history.pop_front();
            }
            self.history.push_back(line.clone());
        }
        line
    }

    fn delete_selection(&mut self) -> bool {
        let Some(range) = self.selected_range() else {
            return false;
        };
        self.buf.replace_range(range.clone(), "");
        self.cursor = range.start;
        self.selection_anchor = None;
        self.history_pos = None;
        self.clear_completion();
        true
    }

    fn clamp_to_boundary(&self, byte: usize) -> usize {
        let mut byte = byte.min(self.buf.len());
        while byte > 0 && !self.buf.is_char_boundary(byte) {
            byte -= 1;
        }
        byte
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordClass {
    Space,
    Word,
    Other,
}

fn word_class(ch: char) -> WordClass {
    if ch.is_whitespace() {
        WordClass::Space
    } else if ch.is_alphanumeric() || ch == '_' {
        WordClass::Word
    } else {
        WordClass::Other
    }
}

fn filtered_text(text: &str) -> String {
    text.chars().filter(|ch| !ch.is_control()).collect()
}

fn prev_boundary(s: &str, cursor: usize) -> usize {
    let mut c = cursor.min(s.len());
    if c == 0 {
        return 0;
    }
    c -= 1;
    while c > 0 && !s.is_char_boundary(c) {
        c -= 1;
    }
    c
}

fn next_boundary(s: &str, cursor: usize) -> usize {
    let mut c = cursor.min(s.len());
    if c >= s.len() {
        return s.len();
    }
    c += 1;
    while c < s.len() && !s.is_char_boundary(c) {
        c += 1;
    }
    c
}

fn char_before(s: &str, cursor: usize) -> Option<(usize, char)> {
    let start = prev_boundary(s, cursor);
    s[start..cursor].chars().next().map(|ch| (start, ch))
}

fn char_at(s: &str, cursor: usize) -> Option<(usize, char)> {
    if cursor >= s.len() {
        return None;
    }
    let end = next_boundary(s, cursor);
    s[cursor..end].chars().next().map(|ch| (end, ch))
}

fn word_left_boundary(s: &str, cursor: usize) -> usize {
    let mut c = cursor.min(s.len());
    while let Some((start, ch)) = char_before(s, c) {
        if word_class(ch) != WordClass::Space {
            break;
        }
        c = start;
    }
    let Some((_, ch)) = char_before(s, c) else {
        return c;
    };
    let class = word_class(ch);
    while let Some((start, ch)) = char_before(s, c) {
        if word_class(ch) != class {
            break;
        }
        c = start;
    }
    c
}

fn word_right_boundary(s: &str, cursor: usize) -> usize {
    let mut c = cursor.min(s.len());
    while let Some((end, ch)) = char_at(s, c) {
        if word_class(ch) != WordClass::Space {
            break;
        }
        c = end;
    }
    let Some((_, ch)) = char_at(s, c) else {
        return c;
    };
    let class = word_class(ch);
    while let Some((end, ch)) = char_at(s, c) {
        if word_class(ch) != class {
            break;
        }
        c = end;
    }
    c
}
