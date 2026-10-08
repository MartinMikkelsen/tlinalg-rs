//! The vendor BLAS/LAPACK library this harness was built against.
//!
//! `tlinalg-blas` deliberately does not know which vendor it links: the host selects it. The
//! harness does have to know, for two reasons.
//!
//! * A row labelled `1T` is only a single-thread measurement if the vendor library also ran on one
//!   thread. LAPACK and BLAS own their threading, so the only way to make that true is to set the
//!   vendor's budget and read it back.
//! * A number is only evidence if the library that produced it can be named. `openblas_get_config`
//!   and friends report what was actually linked and which kernel the runtime dispatched to, which
//!   is not the same as the target the build was configured for.
//!
//! Without a vendor feature the library is absent and every accessor says so; nothing here is
//! conditional in the public API, so the harness does not grow two shapes.

/// Whether a vendor library is linked into this binary.
pub const LINKED: bool = cfg!(any(
    feature = "link-openblas",
    feature = "link-openblas-static"
));

/// How the vendor library was linked: `static`, `shared`, or `none`.
pub fn linkage() -> &'static str {
    if cfg!(feature = "link-openblas-static") {
        "static"
    } else if cfg!(feature = "link-openblas") {
        "shared"
    } else {
        "none"
    }
}

/// What the linked vendor library says about itself.
#[derive(Debug, Clone)]
pub struct Identity {
    /// The vendor library, or `"none"`.
    pub name: &'static str,
    /// [`linkage`] of this binary.
    pub linkage: &'static str,
    /// Version, when the vendor reports one.
    pub version: Option<String>,
    /// The build configuration string, e.g. `OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE`.
    pub config: Option<String>,
    /// The kernel the runtime dispatched to, e.g. `COOPERLAKE`. A Zen host can select an
    /// Intel-named kernel; that is a fact to record, not an error to hide.
    pub corename: Option<String>,
    /// The threading implementation: `sequential`, `pthread` or `openmp`.
    pub parallel: Option<&'static str>,
    /// Processors the vendor library reports.
    pub procs: Option<usize>,
}

impl Identity {
    /// The identity as `key=value` lines, for a manifest to record verbatim.
    pub fn lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("vendor.name", self.name.to_owned()),
            ("vendor.linkage", self.linkage.to_owned()),
            ("vendor.version", self.version.clone().unwrap_or_default()),
            ("vendor.config", self.config.clone().unwrap_or_default()),
            ("vendor.corename", self.corename.clone().unwrap_or_default()),
            (
                "vendor.parallel",
                self.parallel.unwrap_or_default().to_owned(),
            ),
            (
                "vendor.procs",
                self.procs.map_or(String::new(), |p| p.to_string()),
            ),
        ]
    }
}

/// The linked vendor library's identity.
pub fn identity() -> Identity {
    Identity {
        name: if LINKED { "openblas" } else { "none" },
        linkage: linkage(),
        version: sys::config().as_deref().and_then(version_of),
        config: sys::config(),
        corename: sys::corename(),
        parallel: sys::parallel(),
        procs: sys::procs(),
    }
}

/// `OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE` -> `0.3.32`.
fn version_of(config: &str) -> Option<String> {
    let mut fields = config.split_whitespace();
    match (fields.next(), fields.next()) {
        (Some("OpenBLAS"), Some(version)) => Some(version.to_owned()),
        _ => None,
    }
}

/// Set the vendor library's thread budget.
///
/// This is a process-global setting in the vendor library, so the harness sets it once per
/// process from the row's declared budget and re-checks it before each measurement.
pub fn set_threads(threads: usize) -> Result<(), String> {
    if !LINKED {
        return if threads == 1 {
            Ok(())
        } else {
            Err("no vendor library is linked, so a vendor thread budget cannot be set".to_owned())
        };
    }
    sys::set_threads(threads);
    require_threads(threads)
}

/// Check that the vendor library is running on `threads` threads.
///
/// A budget that was requested but not in force would make the row's label a lie, so a mismatch is
/// an error and not a warning.
pub fn require_threads(threads: usize) -> Result<(), String> {
    if !LINKED {
        return Ok(());
    }
    match sys::threads() {
        Some(actual) if actual == threads => Ok(()),
        Some(actual) => Err(format!(
            "openblas reports {actual} threads, but this row declares {threads}"
        )),
        None => Err("openblas does not report its thread count".to_owned()),
    }
}

/// Whether the vendor library supports introspection at all.
pub fn introspectable() -> bool {
    LINKED
}

#[cfg(any(feature = "link-openblas", feature = "link-openblas-static"))]
mod sys {
    use core::ffi::{c_char, c_int, CStr};

    unsafe extern "C" {
        fn openblas_set_num_threads(num_threads: c_int);
        fn openblas_get_num_threads() -> c_int;
        fn openblas_get_num_procs() -> c_int;
        fn openblas_get_config() -> *const c_char;
        fn openblas_get_corename() -> *const c_char;
        fn openblas_get_parallel() -> c_int;
    }

    /// A `char *` the vendor library owns and keeps alive.
    fn string_of(pointer: *const c_char) -> Option<String> {
        if pointer.is_null() {
            return None;
        }
        // SAFETY: the vendor library returns a pointer to a NUL-terminated string that stays valid
        // for the lifetime of the process.
        let text = unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned();
        Some(text)
    }

    pub fn set_threads(threads: usize) {
        // SAFETY: a plain setter on the vendor's global thread count; any positive value is
        // accepted and clamping is the vendor's business.
        unsafe { openblas_set_num_threads(threads as c_int) }
    }

    pub fn threads() -> Option<usize> {
        // SAFETY: no arguments, no state touched beyond reading the vendor's thread count.
        let threads = unsafe { openblas_get_num_threads() };
        (threads > 0).then_some(threads as usize)
    }

    pub fn procs() -> Option<usize> {
        // SAFETY: as `threads`.
        let procs = unsafe { openblas_get_num_procs() };
        (procs > 0).then_some(procs as usize)
    }

    pub fn config() -> Option<String> {
        // SAFETY: returns a pointer to a static string owned by the vendor library.
        string_of(unsafe { openblas_get_config() })
    }

    pub fn corename() -> Option<String> {
        // SAFETY: as `config`.
        string_of(unsafe { openblas_get_corename() })
    }

    pub fn parallel() -> Option<&'static str> {
        // SAFETY: no arguments, returns an enumerator: 0 sequential, 1 pthread, 2 openmp.
        match unsafe { openblas_get_parallel() } {
            0 => Some("sequential"),
            1 => Some("pthread"),
            2 => Some("openmp"),
            _ => None,
        }
    }
}

#[cfg(not(any(feature = "link-openblas", feature = "link-openblas-static")))]
mod sys {
    pub fn set_threads(_threads: usize) {}
    pub fn threads() -> Option<usize> {
        None
    }
    pub fn procs() -> Option<usize> {
        None
    }
    pub fn config() -> Option<String> {
        None
    }
    pub fn corename() -> Option<String> {
        None
    }
    pub fn parallel() -> Option<&'static str> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_version_is_read_out_of_the_config_string() {
        assert_eq!(
            version_of("OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE").as_deref(),
            Some("0.3.32")
        );
        assert_eq!(version_of("something else"), None);
    }

    /// The point of the module: a budget that was set is the budget in force, and it can be read
    /// back. Without a vendor feature this only asserts the feature-independent behaviour.
    #[test]
    fn the_vendor_budget_is_set_and_read_back() {
        if !LINKED {
            assert!(set_threads(4).is_err(), "no vendor library to set");
            assert!(require_threads(4).is_ok(), "nothing to contradict");
            return;
        }
        for threads in [1, 3, 8] {
            set_threads(threads).expect("the vendor accepts a thread budget");
            assert_eq!(sys::threads(), Some(threads));
        }
        set_threads(1).unwrap();
        let identity = identity();
        assert_eq!(identity.name, "openblas");
        assert!(
            identity.config.is_some(),
            "the vendor must be able to name itself"
        );
        println!(
            "{}",
            identity
                .lines()
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}
