//! Cumulative sums that restart every `cell_size` points.
//!
//! Plain prefix sums grow with `n`, which loses precision when a small cluster
//! cost is obtained as a difference of two large numbers. Restarting the
//! prefix sums in cells keeps the numbers small. A cluster may touch at most
//! two cells, so `cell_size` must be at least `max_cluster_len - 1`.

use crate::cost::sse_from_sums;

pub struct CellSums {
    cell_size: usize,
    /// row-major `num_cells x (cell_size + 1)`
    sum: Vec<f64>,
    sum2: Vec<f64>,
}

impl CellSums {
    pub fn new(v: &[f64], cell_size: usize) -> Self {
        let num_cells = v.len().div_ceil(cell_size);
        let w = cell_size + 1;
        let mut sum = vec![0.0; num_cells * w];
        let mut sum2 = vec![0.0; num_cells * w];
        for c in 0..num_cells {
            let off = c * cell_size;
            let len = (v.len() - off).min(cell_size);
            for j in 0..len {
                let x = v[off + j];
                sum[c * w + j + 1] = sum[c * w + j] + x;
                let x2 = x * x;
                sum2[c * w + j + 1] = sum2[c * w + j] + x2;
            }
        }
        Self { cell_size, sum, sum2 }
    }

    /// SSE of the points `i..=j` (both inclusive); `0` if `j <= i`.
    #[inline]
    pub fn sse(&self, i: usize, j: usize) -> f64 {
        let cs = self.cell_size;
        let w = cs + 1;
        let (ci, ri) = (i / cs, i % cs);
        let (cj, rj) = (j / cs, j % cs);
        if ci == cj {
            if rj <= ri {
                return 0.0;
            }
            let b = ci * w;
            sse_from_sums(
                self.sum[b + rj + 1] - self.sum[b + ri],
                self.sum2[b + rj + 1] - self.sum2[b + ri],
                rj + 1 - ri,
            )
        } else {
            debug_assert!(cj == ci + 1, "cluster spans more than two cells");
            let (bi, bj) = (ci * w, cj * w);
            let i_cs = self.sum[bi + cs] - self.sum[bi + ri];
            let i_cs2 = self.sum2[bi + cs] - self.sum2[bi + ri];
            let j_cs = self.sum[bj + rj + 1];
            let j_cs2 = self.sum2[bj + rj + 1];
            let val1 = j_cs + i_cs;
            let mu = val1 / (rj + 1 + cs - ri) as f64;
            let mut result = j_cs2 + i_cs2;
            result -= (2.0 * mu) * val1;
            result += (rj as f64 - ri as f64 + 1.0 + cs as f64) * (mu * mu);
            result.max(0.0)
        }
    }
}

/// Same values (bit for bit) as [`CellSums`], but laid out by global position so a
/// lookup needs no row arithmetic and touches few cache lines. About 20-30% faster
/// for the DP methods that evaluate `i -> j` edges in unpredictable order (slower
/// than [`CellSums`] for the sequential scans of the simple DP).
pub struct IndexedCellSums {
    cell_size: usize,
    /// `[sum[r], sum2[r]]`: in-cell prefix sums *before* the position
    lo: Vec<[f64; 2]>,
    /// `[sum[r + 1], sum2[r + 1]]`: in-cell prefix sums *including* the position
    hi: Vec<[f64; 2]>,
    /// totals `[sum, sum2]` of the whole cell containing the position
    tot: Vec<[f64; 2]>,
}

impl IndexedCellSums {
    pub fn new(v: &[f64], cell_size: usize) -> Self {
        let n = v.len();
        let mut lo = vec![[0.0; 2]; n];
        let mut hi = vec![[0.0; 2]; n];
        let mut tot = vec![[0.0; 2]; n];
        let mut off = 0;
        while off < n {
            let len = (n - off).min(cell_size);
            let (mut s, mut s2) = (0.0, 0.0);
            for j in 0..len {
                lo[off + j] = [s, s2];
                let x = v[off + j];
                s += x;
                s2 += x * x;
                hi[off + j] = [s, s2];
            }
            // totals are only read for the first cell of a two-cell cluster (always a full cell)
            for t in &mut tot[off..off + len] {
                *t = [s, s2];
            }
            off += cell_size;
        }
        Self { cell_size, lo, hi, tot }
    }

    /// SSE of the points `i..=j` (both inclusive); `0` if `j <= i`.
    #[inline]
    pub fn sse(&self, i: usize, j: usize) -> f64 {
        let cs = self.cell_size;
        if i / cs == j / cs {
            if j <= i {
                return 0.0;
            }
            let (a, b) = (&self.lo[i], &self.hi[j]);
            sse_from_sums(b[0] - a[0], b[1] - a[1], j + 1 - i)
        } else {
            debug_assert!(j / cs == i / cs + 1, "cluster spans more than two cells");
            let (a, b, t) = (&self.lo[i], &self.hi[j], &self.tot[i]);
            let i_cs = t[0] - a[0];
            let i_cs2 = t[1] - a[1];
            let val1 = b[0] + i_cs;
            let cnt = (j + 1 - i) as f64;
            let mu = val1 / cnt;
            let mut result = b[1] + i_cs2;
            result -= (2.0 * mu) * val1;
            result += cnt * (mu * mu);
            result.max(0.0)
        }
    }
}
