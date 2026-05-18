//! Background job system: a `rayon` worker pool plus a `crossbeam-channel`
//! result stream.
//!
//! Chunk generation, lighting recomputes, and meshing are all CPU-bound and
//! pure with respect to the chunk they touch, so they ship off to worker
//! threads. The main thread drains [`JobResult`](crate::jobs::JobResult)
//! values once per frame and never blocks on a worker.
