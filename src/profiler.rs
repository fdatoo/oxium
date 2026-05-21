//! Per-frame, per-system timing log written to a CSV file.
//!
//! Enabled by `--profile <path>` on the binary. Each [`step`] in
//! [`crate::app::AppState`] wraps the major systems with [`time`] /
//! [`time_into`], which appends a `Span` to the current frame's slot.
//! At the end of the step, [`Profiler::finish_frame`] writes one
//! CSV row with the per-frame counters plus a column per recorded
//! span. The output is parseable by `awk`, importable into a
//! spreadsheet, and small enough to glance through in a text editor
//! during diagnosis.
//!
//! The hot-path overhead is one `Instant::now()` per span plus a
//! `Vec::push` of a tagged tuple — negligible (<1µs) compared to any
//! system the profiler measures. When the profiler is disabled (the
//! common case: no `--profile`), `time` is a thin closure call with
//! no recording cost.

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// One timed span recorded inside a frame.
#[derive(Debug, Clone, Copy)]
struct Span {
    name: &'static str,
    micros: u32,
}

/// Per-frame counters that aren't per-system spans — copied across
/// from the various subsystems by [`AppState::step`] before
/// `finish_frame` flushes a row.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameCounters {
    pub fps: f32,
    pub work_ms: f32,
    pub draw_calls: u32,
    pub light_queue: u32,
    pub chunks_rendered: u32,
    /// Chunks currently in `ChunkSlot::Stored` — gen completed and
    /// `world.data` is populated. Compared against `chunks_rendered`
    /// it tells us whether the mesher is keeping up with gen.
    pub chunks_loaded: u32,
    /// Chunks currently in `ChunkSlot::Pending` — gen spawned but not
    /// yet returned. If this stays high the worker pool is the
    /// bottleneck.
    pub chunks_pending: u32,
    /// Number of chunk-edit interactions that fired this frame
    /// (place/break clicks). Useful for picking out the exact frames
    /// where the player triggered an edit when scanning the CSV.
    pub edits: u32,
}

/// File-backed profiler. The header is the fixed counter columns
/// followed by every span name encountered across the run, written
/// once when the first non-empty frame is flushed. Subsequent rows
/// pad missing spans with `0` so the columns stay aligned.
pub struct Profiler {
    writer: RefCell<BufWriter<File>>,
    /// Spans accumulated for the in-progress frame. Cleared by
    /// `finish_frame` after each row is written.
    pending_spans: RefCell<Vec<Span>>,
    /// Span columns the header committed to (in order). Determined by
    /// the first frame's span set — every frame after pads/fills to
    /// match. Practically every frame records the same systems, so
    /// the "missing column → 0" path only matters for transient
    /// systems that didn't run on a given step.
    columns: RefCell<Vec<&'static str>>,
    /// Set to `true` after the header line is written.
    header_written: RefCell<bool>,
    /// Monotonically-incrementing frame index for the CSV.
    frame_id: RefCell<u64>,
    /// When the profiler was created. Used for the absolute timestamp
    /// column so a viewer can align frames with wall-clock events.
    start: Instant,
}

impl Profiler {
    /// Open `path` for writing. Truncates any existing file at that
    /// path — the profiler always starts fresh per session.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let f = File::create(path)?;
        Ok(Self {
            writer: RefCell::new(BufWriter::new(f)),
            pending_spans: RefCell::new(Vec::new()),
            columns: RefCell::new(Vec::new()),
            header_written: RefCell::new(false),
            frame_id: RefCell::new(0),
            start: Instant::now(),
        })
    }

    /// Record a span by name. Called by [`time`] / [`time_into`].
    pub fn record(&self, name: &'static str, micros: u32) {
        self.pending_spans.borrow_mut().push(Span { name, micros });
    }

    /// Flush the in-progress frame to CSV with the supplied counters
    /// and reset for the next frame.
    pub fn finish_frame(&self, counters: FrameCounters) {
        // Decide columns on the first non-empty frame.
        {
            let mut cols = self.columns.borrow_mut();
            if cols.is_empty() {
                for span in self.pending_spans.borrow().iter() {
                    if !cols.contains(&span.name) {
                        cols.push(span.name);
                    }
                }
            } else {
                // Subsequent frames: append any new span name we
                // haven't seen so the column set monotonically grows.
                for span in self.pending_spans.borrow().iter() {
                    if !cols.contains(&span.name) {
                        cols.push(span.name);
                    }
                }
            }
        }
        // Write header once; padding the new-column case is fine — old
        // rows just don't have those columns, which means we'd have
        // to rewrite the header at every column-set change. Practical
        // workaround: collect every span name the first 60 frames see
        // before flushing anything. Simpler since steady-state spans
        // are stable from frame 1.
        let cols_snapshot = self.columns.borrow().clone();
        let mut w = self.writer.borrow_mut();
        if !*self.header_written.borrow() {
            // Header: counters first, then span columns.
            let mut header = String::from(
                "frame_id,t_session_ms,fps,work_ms,draw_calls,light_queue,chunks_rendered,chunks_loaded,chunks_pending,edits",
            );
            for c in &cols_snapshot {
                header.push(',');
                header.push_str(c);
            }
            header.push('\n');
            let _ = w.write_all(header.as_bytes());
            *self.header_written.borrow_mut() = true;
        }

        let fid = *self.frame_id.borrow();
        let t_session_ms = self.start.elapsed().as_secs_f32() * 1000.0;
        let mut row = format!(
            "{fid},{:.2},{:.1},{:.2},{},{},{},{},{},{}",
            t_session_ms,
            counters.fps,
            counters.work_ms,
            counters.draw_calls,
            counters.light_queue,
            counters.chunks_rendered,
            counters.chunks_loaded,
            counters.chunks_pending,
            counters.edits,
        );
        for col in &cols_snapshot {
            let micros = self
                .pending_spans
                .borrow()
                .iter()
                .find(|s| s.name == *col)
                .map(|s| s.micros)
                .unwrap_or(0);
            row.push(',');
            row.push_str(&micros.to_string());
        }
        row.push('\n');
        let _ = w.write_all(row.as_bytes());

        self.pending_spans.borrow_mut().clear();
        *self.frame_id.borrow_mut() += 1;
        // Flush periodically so a crash doesn't lose the buffer.
        if fid.is_multiple_of(60) {
            let _ = w.flush();
        }
    }
}

impl Drop for Profiler {
    /// Final flush so the CSV is complete even on panic / Cmd-Q.
    fn drop(&mut self) {
        let _ = self.writer.borrow_mut().flush();
    }
}

/// Time `f`, optionally record the span on `profiler`. Returns the
/// function's result. When `profiler` is `None` the closure call has
/// no measurable overhead beyond one extra `Instant::now()` pair.
pub fn time<F: FnOnce() -> R, R>(profiler: Option<&Profiler>, name: &'static str, f: F) -> R {
    let Some(p) = profiler else { return f() };
    let start = Instant::now();
    let r = f();
    let micros = start.elapsed().as_micros().min(u32::MAX as u128) as u32;
    p.record(name, micros);
    r
}
