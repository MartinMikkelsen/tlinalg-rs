//! Case parsing and deterministic fixture selection.

use std::fmt;

/// A requested matrix shape and batch count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Case {
    /// Matrix rows.
    pub m: usize,
    /// Matrix columns.
    pub n: usize,
    /// Number of matrices.
    pub batch: usize,
}

impl Case {
    /// Stable label used in diagnostics.
    pub fn label(self) -> String {
        format!("{}x{}xb{}", self.m, self.n, self.batch)
    }
}

/// All operation families in the harness.
pub const FAMILIES: &[&str] = &[
    "lu_factor",
    "lu_solve_prepared",
    "lu_factor_solve",
    "solve",
    "lu",
    "cholesky",
    "triangular_solve",
    "qr",
    "rank_revealing_qr",
    "svd_thin",
    "svd_full",
    "svd_values",
    "eigh",
    "eigvalsh",
    "eig",
    "eigvals",
    "full_piv_lu",
    "full_piv_lu_solve",
    "householder_factor",
    "householder_apply",
];

/// Parse `N` or `MxN`.
pub fn parse_shape(text: &str) -> Result<(usize, usize), String> {
    let mut parts = text.split('x');
    let m = parts
        .next()
        .ok_or_else(|| format!("invalid shape {text:?}"))?
        .parse::<usize>()
        .map_err(|_| format!("invalid shape {text:?}"))?;
    let n = match parts.next() {
        None => m,
        Some(value) if parts.next().is_none() => value
            .parse()
            .map_err(|_| format!("invalid shape {text:?}"))?,
        _ => return Err(format!("invalid shape {text:?}")),
    };
    if m == 0 || n == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    Ok((m, n))
}

/// Parse a comma-separated positive integer list.
pub fn parse_usizes(text: &str, what: &str) -> Result<Vec<usize>, String> {
    let values: Result<Vec<_>, _> = text.split(',').map(|v| v.parse::<usize>()).collect();
    let values = values.map_err(|_| format!("invalid {what} list {text:?}"))?;
    if values.is_empty() || values.contains(&0) {
        return Err(format!("{what} values must be positive"));
    }
    Ok(values)
}

/// Whether a family accepts this shape.
pub fn applicable(family: &str, c: Case) -> bool {
    match family {
        "cholesky" | "lu_factor" | "lu_solve_prepared" | "lu_factor_solve" | "solve" | "lu"
        | "triangular_solve" | "eigh" | "eigvalsh" | "eig" | "eigvals" | "full_piv_lu"
        | "full_piv_lu_solve" => c.m == c.n,
        "householder_factor" | "householder_apply" | "qr" | "rank_revealing_qr" | "svd_thin"
        | "svd_full" | "svd_values" => true,
        _ => false,
    }
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}
