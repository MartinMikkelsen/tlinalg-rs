//! Wall-clock priming and repetition policy.

use std::time::{Duration, Instant};

/// Result of one measured row.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// Best complete operation time in milliseconds.
    pub best_ms: f64,
}

/// Prime for a wall-clock duration, then take the minimum of `reps` operations.
///
/// `prepare` is deliberately called outside each clocked region: fixture restoration is host work,
/// not provider work (unlike the old Criterion closures, which restored inside their boundary).
pub fn measure(
    prime_ms: u64,
    reps: usize,
    mut prepare: impl FnMut() -> Result<(), String>,
    mut operation: impl FnMut() -> Result<(), String>,
) -> Result<Timing, String> {
    let until = Instant::now() + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        prepare()?;
        operation()?;
    }
    let mut best = Duration::MAX;
    for _ in 0..reps.max(1) {
        prepare()?;
        let start = Instant::now();
        operation()?;
        best = best.min(start.elapsed());
    }
    Ok(Timing {
        best_ms: best.as_secs_f64() * 1_000.0,
    })
}
