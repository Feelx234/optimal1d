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
