//! Cost functions.
//!
//! A [`Segment`] is a plain cluster cost `seg(i, j)` of the sorted points
//! `i..j` (upper exclusive, `j > i`). Algorithms that solve the DP via
//! Monge/concave arguments use the [`DpCost`] trait, which additionally folds in
//! the DP values `F[i]` and, for microaggregation, the cluster size constraint
//! `k <= j - i <= 2k - 1`.

use crate::stable::CellSums;

/// Cost of the cluster consisting of the sorted points `i..j` (`i < j`).
pub trait Segment {
    fn seg(&self, i: usize, j: usize) -> f64;
}

impl<S: Segment + ?Sized> Segment for &S {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        (**self).seg(i, j)
    }
}

/// Cost of an edge `i -> j` in the DP graph including the DP value `fv[i]`.
pub trait DpCost {
    fn calc(&self, i: usize, j: usize, fv: &[f64]) -> f64;
}

pub fn cumsum(v: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(v.len() + 1);
    let mut s = 0.0;
    out.push(s);
    for &x in v {
        s += x;
        out.push(s);
    }
    out
}

/// `sum_sq - sum^2 / n`, clamped at 0 to avoid numerical issues.
#[inline(always)]
pub fn sse_from_sums(sum: f64, sum_sq: f64, n: usize) -> f64 {
    let mut result = sum_sq;
    result -= sum / n as f64 * sum;
    result.max(0.0)
}

// ---------------------------------------------------------------- SSE

/// SSE via plain cumulative sums.
pub struct SseSegment {
    pub cumsum: Vec<f64>,
    pub cumsum2: Vec<f64>,
}
impl SseSegment {
    pub fn new(v: &[f64]) -> Self {
        let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
        Self { cumsum: cumsum(v), cumsum2: cumsum(&sq) }
    }
}
impl Segment for SseSegment {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        sse_from_sums(
            self.cumsum[j] - self.cumsum[i],
            self.cumsum2[j] - self.cumsum2[i],
            j - i,
        )
    }
}

/// Same as [`FasterSseSegment`] but evaluated as `-n * mean^2` (the form used
/// by the adapted microaggregation cost, differs in rounding).
pub struct FasterSseMeanSegment {
    pub cumsum: Vec<f64>,
}
impl FasterSseMeanSegment {
    pub fn new(v: &[f64]) -> Self {
        Self { cumsum: cumsum(v) }
    }
}
impl Segment for FasterSseMeanSegment {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        let n = (j - i) as f64;
        let x_bar = (self.cumsum[j] - self.cumsum[i]) / n;
        -n * x_bar * x_bar
    }
}

/// SSE where the cumulative sums are restarted every `cell_size` points
/// (more numerically stable).
pub struct StableSseSegment {
    pub cells: CellSums,
}
impl StableSseSegment {
    pub fn new(v: &[f64], cell_size: usize) -> Self {
        Self { cells: CellSums::new(v, cell_size) }
    }
}
impl Segment for StableSseSegment {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        self.cells.sse(i, j - 1)
    }
}

/// `-(sum^2)/n`: SSE up to a constant (the sum of squares), cheaper to compute.
pub struct FasterSseSegment {
    pub cumsum: Vec<f64>,
}
impl FasterSseSegment {
    pub fn new(v: &[f64]) -> Self {
        Self { cumsum: cumsum(v) }
    }
}
impl Segment for FasterSseSegment {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        let n = (j - i) as f64;
        let delta = self.cumsum[j] - self.cumsum[i];
        -delta * delta / n
    }
}

/// SSE computed directly (two passes) for each cluster. Slow but exact.
pub struct NoPrecomputeSseSegment<'a> {
    pub v: &'a [f64],
}
impl Segment for NoPrecomputeSseSegment<'_> {
    fn seg(&self, i: usize, j: usize) -> f64 {
        let s = &self.v[i..j];
        let mut mean = 0.0;
        for &x in s {
            mean += x;
        }
        mean /= s.len() as f64;
        let mut r = 0.0;
        for &x in s {
            r += (x - mean) * (x - mean);
        }
        r
    }
}

// -------------------------------------------------- other cost functions

/// Sum of absolute errors to the median.
pub struct SaeSegment {
    pub cumsum: Vec<f64>,
}
impl SaeSegment {
    pub fn new(v: &[f64]) -> Self {
        Self { cumsum: cumsum(v) }
    }
}
impl Segment for SaeSegment {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        let c = &self.cumsum;
        let center = (j - i) / 2;
        c[j] - c[j - center] - (c[i + center] - c[i])
    }
}

/// Maximum distance inside a cluster (`x[j-1] - x[i]`).
pub struct MaxDistSegment<'a> {
    pub v: &'a [f64],
}
impl Segment for MaxDistSegment<'_> {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        self.v[j - 1] - self.v[i]
    }
}

/// Cost of rounding every point up to the cluster maximum.
pub struct RoundUpSegment<'a> {
    pub v: &'a [f64],
    pub cumsum: Vec<f64>,
}
impl<'a> RoundUpSegment<'a> {
    pub fn new(v: &'a [f64]) -> Self {
        Self { v, cumsum: cumsum(v) }
    }
}
impl Segment for RoundUpSegment<'_> {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        (j - i) as f64 * self.v[j - 1] - (self.cumsum[j] - self.cumsum[i])
    }
}

/// Cost of rounding every point down to the cluster minimum.
pub struct RoundDownSegment<'a> {
    pub v: &'a [f64],
    pub cumsum: Vec<f64>,
}
impl<'a> RoundDownSegment<'a> {
    pub fn new(v: &'a [f64]) -> Self {
        Self { v, cumsum: cumsum(v) }
    }
}
impl Segment for RoundDownSegment<'_> {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        self.cumsum[j] - self.cumsum[i] - (j - i) as f64 * self.v[i]
    }
}

// ----------------------------------------------------- DP cost adapters

/// Microaggregation adaptation of a [`Segment`] cost.
///
/// Edges with a cluster size outside `k..=2k-1` are given costs that keep the
/// cost matrix concave Monge but are larger than any feasible solution:
/// `LARGE -/+ SMALL * i`.
pub struct Adapted<S> {
    pub seg: S,
    pub k: usize,
    pub small: f64,
    pub large: f64,
}
impl<S> Adapted<S> {
    pub fn new(seg: S, k: usize, small: f64, large: f64) -> Self {
        Self { seg, k, small, large }
    }
}
impl<S: Segment> DpCost for Adapted<S> {
    #[inline]
    fn calc(&self, i: usize, j: usize, fv: &[f64]) -> f64 {
        if j <= i {
            return f64::INFINITY;
        }
        if j - i < self.k {
            return self.large + self.small * i as f64;
        }
        if j - i > 2 * self.k - 1 {
            return self.large - self.small * i as f64;
        }
        self.seg.seg(i, j) + fv[i]
    }
}

/// Regularized (penalty `lambda` per cluster) k-means cost, no size limits.
pub struct Regularized<S> {
    pub seg: S,
    pub lambda: f64,
}
impl<S: Segment> DpCost for Regularized<S> {
    #[inline]
    fn calc(&self, i: usize, j: usize, fv: &[f64]) -> f64 {
        if j <= i {
            return f64::INFINITY;
        }
        self.seg.seg(i, j) + self.lambda + fv[i]
    }
}
