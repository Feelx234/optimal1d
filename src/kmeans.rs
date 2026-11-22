//! Optimal 1d k-means: Gronlund et al. (2017), "Fast Exact k-Means, k-Medians and
//! Bregman Divergence Clustering in 1D", and Wu's matrix searching DP.

use crate::clustering::{implicit_to_explicit, num_clusters};
use crate::concave::wilber_fast1dkmeans as wilber;
use crate::cost::{cumsum, Regularized, Segment};
use crate::smawk::{smawk, Scratch};

/// Squared error of the points `i..j` around their mean, computed from prefix sums.
pub struct KmeansSegment {
    pub cumsum: Vec<f64>,
    pub cumsum2: Vec<f64>,
}

impl KmeansSegment {
    pub fn new(v: &[f64]) -> Self {
        let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
        Self { cumsum: cumsum(v), cumsum2: cumsum(&sq) }
    }

    /// Cost of the points `i..=j` (both inclusive); 0 if `j <= i`.
    #[inline]
    pub fn inclusive(&self, i: usize, j: usize) -> f64 {
        if j <= i {
            return 0.0;
        }
        self.seg(i, j + 1)
    }
}

impl Segment for KmeansSegment {
    #[inline]
    fn seg(&self, i: usize, j: usize) -> f64 {
        if j <= i + 1 {
            return 0.0;
        }
        let n = (j - i) as f64;
        let s = self.cumsum[j] - self.cumsum[i];
        let mu = s / n;
        let mut result = self.cumsum2[j] - self.cumsum2[i];
        result += n * (mu * mu);
        result -= (2.0 * mu) * s;
        result.max(0.0)
    }
}

/// Solves the lambda-regularized problem (cost + lambda per cluster);
/// returns the implicit assignment.
pub fn regularized(seg: &KmeansSegment, n: usize, lambda: f64) -> Vec<usize> {
    wilber(n, &Regularized { seg, lambda })
}

/// Cost of the clustering given by an implicit assignment.
pub fn implicit_cost(result: &[usize], seg: &KmeansSegment) -> f64 {
    let mut cost = 0.0;
    let mut pos = result.len() - 1;
    while result[pos] > 0 {
        cost += seg.inclusive(result[pos], pos);
        pos = result[pos] - 1;
    }
    cost + seg.inclusive(0, pos)
}

/// Optimal k-means of the sorted `v` by binary search over the regularization
/// strength. `interpolation` selects the faster variant of section 5.2.
pub fn binary_search(v: &[f64], k: usize, max_iter: usize, epsilon: f64, interpolation: bool) -> Vec<i64> {
    let n = v.len();
    if k == n {
        return (0..n as i64).collect();
    }
    let seg = KmeansSegment::new(v);
    let (mut l_low, mut k_low, mut c_low) = (0.0f64, n, 0.0f64);
    let mut l_high = seg.inclusive(0, n - 1);
    let (mut k_high, mut c_high) = (1usize, l_high);

    let mut result = vec![0usize; n];
    for _ in 0..max_iter {
        let mut lambda_mid = (l_high + l_low) / 2.0;
        if interpolation {
            let lambda_test = (c_high - c_low) / (k_low - k_high) as f64;
            if lambda_test >= l_low && lambda_test <= l_high {
                lambda_mid = lambda_test;
            }
        }
        result = regularized(&seg, n, lambda_mid);
        let k_mid = num_clusters(&result);
        if k_mid == k {
            if l_high - l_low < epsilon {
                break;
            }
            l_high = lambda_mid;
            k_high = k_mid;
        } else if k_mid < k {
            // too few clusters, decrease lambda
            l_high = lambda_mid;
            k_high = k_mid;
            if interpolation {
                c_high = implicit_cost(&result, &seg);
            }
        } else {
            // too many clusters, increase lambda
            l_low = lambda_mid;
            k_low = k_mid;
            if interpolation {
                c_low = implicit_cost(&result, &seg);
            }
        }
    }
    implicit_to_explicit(&result)
}

/// Cost matrix of Wu's DP: row `i` (end of the last cluster, inclusive), column
/// `j` (its start). `d` holds the optimal costs for one cluster less.
fn xi_cost(seg: &KmeansSegment, d: &[f64], i: usize, j: usize) -> f64 {
    let n = d.len();
    // index of the best (k-1)-clustering of the points before j; `j == 0`
    // intentionally wraps around to the last entry (a prohibitively large value)
    let idx = if j == 0 {
        n - 1
    } else if i + 1 < j {
        i
    } else {
        j - 1
    };
    d[idx] + seg.inclusive(j, i)
}

/// O(kn) time and space optimal k-means (Wu, "Optimal quantization by matrix searching").
pub fn cluster_xi(v: &[f64], k: usize) -> Vec<i64> {
    let n = v.len();
    let seg = KmeansSegment::new(v);
    let mut d: Vec<Vec<f64>> = vec![(0..n).map(|j| seg.inclusive(0, j)).collect(), vec![0.0; n]];
    let mut t = vec![vec![0usize; n]; k];
    let mut scratch = Scratch::new();
    for kk in 1..k {
        let (cur, next) = ((kk - 1) % 2, kk % 2);
        let mut argmin = vec![0usize; n];
        {
            let dr = &d[cur];
            smawk(0, n, 0, n, &|i, j| xi_cost(&seg, dr, i, j), &mut argmin, &mut scratch);
        }
        for i in 0..n {
            d[next][i] = xi_cost(&seg, &d[cur], i, argmin[i]);
        }
        t[kk] = argmin;
    }
    // back tracking
    let mut out = vec![0i64; n];
    let mut start = n;
    for kk in (0..k).rev() {
        let stop = start;
        // like the original (numpy negative indexing) `start == 0` wraps to the last entry
        start = t[kk][if start == 0 { n - 1 } else { start - 1 }];
        // an inverted range is empty (as `range(start, stop)` in the original)
        if start < stop {
            for o in &mut out[start..stop] {
                *o = kk as i64;
            }
        }
    }
    out
}

/// Same as [`cluster_xi`] in O(n) space: determines the regularization that yields
/// exactly `k` clusters from the last two DP rows and solves that problem.
pub fn cluster_xi_space(v: &[f64], k: usize) -> Result<Vec<i64>, &'static str> {
    let n = v.len();
    if k == 1 {
        return Ok(vec![0; n]);
    }
    if k == n {
        return Ok((0..n as i64).collect());
    }
    let seg = KmeansSegment::new(v);
    let mut d: Vec<Vec<f64>> = vec![(0..n).map(|j| seg.inclusive(0, j)).collect(), vec![0.0; n]];
    let mut optimal = vec![0usize; n];
    let mut scratch = Scratch::new();
    let (mut cur, mut next) = (0, 0);
    for kk in 1..=k {
        cur = (kk - 1) % 2;
        next = kk % 2;
        {
            let dr = &d[cur];
            smawk(kk, n, 0, n, &|i, j| xi_cost(&seg, dr, i, j), &mut optimal, &mut scratch);
        }
        for i in 0..n {
            d[next][i] = xi_cost(&seg, &d[cur], i, optimal[i]);
        }
    }
    let lambda = d[cur][n - 1] - d[next][n - 1];
    if lambda < 0.0 {
        return Err("using more clusters should not increase the cost");
    }
    Ok(implicit_to_explicit(&regularized(&seg, n, lambda)))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64) / ((1u64 << 31) as f64)
        }
    }

    /// Optimal cost of k clusters by the plain O(k n^2) DP.
    fn brute_cost(v: &[f64], k: usize) -> f64 {
        let n = v.len();
        let seg = KmeansSegment::new(v);
        let mut d = vec![f64::INFINITY; n + 1];
        d[0] = 0.0;
        for _ in 0..k {
            let mut nd = vec![f64::INFINITY; n + 1];
            for j in 1..=n {
                for i in 0..j {
                    nd[j] = nd[j].min(d[i] + seg.seg(i, j));
                }
            }
            d = nd;
        }
        d[n]
    }

    fn labels_cost(v: &[f64], labels: &[i64]) -> f64 {
        let k = *labels.iter().max().unwrap() as usize + 1;
        let mut total = 0.0;
        for l in 0..k {
            let pts: Vec<f64> = v.iter().zip(labels).filter(|(_, &g)| g == l as i64).map(|(x, _)| *x).collect();
            if pts.is_empty() {
                continue; // labels may skip a value when duplicates make clusters redundant
            }
            let m = pts.iter().sum::<f64>() / pts.len() as f64;
            total += pts.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
        }
        total
    }

    fn datasets() -> Vec<Vec<f64>> {
        let mut rng = Lcg(42);
        let mut out = Vec::new();
        for n in 1..=40usize {
            for kind in 0..4 {
                let mut v: Vec<f64> = (0..n)
                    .map(|_| match kind {
                        0 => rng.next(),
                        1 => (rng.next() * 5.0).floor(),
                        2 => 1e3 + rng.next(),
                        _ => [0.0, 10.0, 50.0][(rng.next() * 3.0) as usize] + rng.next() * 0.1,
                    })
                    .collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                out.push(v);
            }
        }
        out
    }

    #[test]
    fn all_methods_optimal_vs_brute() {
        for v in datasets() {
            let n = v.len();
            for k in 1..=n {
                let opt = brute_cost(&v, k);
                let tol = 1e-6 * (1.0 + opt) + 1e-9 * v.iter().map(|x| x * x).sum::<f64>();
                let results = [
                    ("kn", cluster_xi(&v, k)),
                    ("space", cluster_xi_space(&v, k).unwrap()),
                    ("bs0", binary_search(&v, k, 200, 1e-10, false)),
                    ("bs1", binary_search(&v, k, 200, 1e-10, true)),
                ];
                for (name, labels) in results {
                    assert_eq!(labels.len(), n);
                    let nc = *labels.iter().max().unwrap() as usize + 1;
                    // binary search may land on a tie with fewer/more clusters when
                    // the data has duplicates; only compare cost when k matches
                    if nc == k {
                        let c = labels_cost(&v, &labels);
                        assert!((c - opt).abs() <= tol, "{name} n={n} k={k}: {c} vs {opt} for {v:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn smawk_dp_matches_brute() {
        // the SMAWK based DP (cluster_xi) must hit exactly k non-empty clusters at optimal cost
        for v in datasets() {
            let n = v.len();
            for k in 1..=n {
                let labels = cluster_xi(&v, k);
                let distinct = *labels.iter().max().unwrap() as usize + 1;
                assert_eq!(distinct, k, "n={n} k={k} {v:?}");
                let opt = brute_cost(&v, k);
                assert!((labels_cost(&v, &labels) - opt).abs() <= 1e-6 * (1.0 + opt), "n={n} k={k} {v:?} {labels:?} {} {opt}", labels_cost(&v, &labels));
            }
        }
    }
}
