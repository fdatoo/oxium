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
