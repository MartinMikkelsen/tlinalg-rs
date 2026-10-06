//! Library-owned Auto batch lane selection.

// Default policy adapted from tenferro-rs 527f58cd,
// crates/tenferro-linalg/src/cpu/tlinalg.rs:42-115 and
// crates/tenferro-cpu/src/batch_policy.rs:78-97,312-316 (MIT OR Apache-2.0).
// ponytail: retain the existing dimension cutoff; work-model calibration belongs here later.
const AUTO_FAN_OUT_MAX_ITEM_DIM: usize = 64;

/// Upper bound on contiguous chunks; the resource itself stays in the call's token.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LanePlan {
    pub(crate) lanes: usize,
}

impl LanePlan {
    /// `None` preserves packed LU's existing no-size-cutoff policy.
    pub(crate) fn resolve(batch: usize, width: usize, max_item_dim: Option<usize>) -> Self {
        let fans_out = width > 1
            && batch >= width
            && max_item_dim.is_none_or(|dim| dim <= AUTO_FAN_OUT_MAX_ITEM_DIM);
        Self {
            lanes: if fans_out { width } else { 1 },
        }
    }
}
