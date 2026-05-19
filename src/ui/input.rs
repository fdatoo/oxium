//! Input routing decisions. `Ui::on_key` and friends are defined on
//! `Ui` itself (see `mod.rs`); this file just hosts the small shared
//! types.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputDisposition {
    /// The UI handled the event. `main.rs` does not forward it to
    /// `InputBuf`.
    Consumed,
    /// The UI didn't care. `main.rs` forwards to `InputBuf` as today.
    Forward,
}
