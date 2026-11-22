//! Optimal univariate microaggregation: partition sorted points into
//! contiguous clusters of at least `k` points minimizing a cost.

use crate::clustering::{implicit_to_explicit, trivial_microagg};
use crate::concave::{galil_park, simple_concave, staggered, wilber};
use crate::cost::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Simple,
    Wilber,
    GalilPark,
    Staggered,
}

impl Method {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "simple" => Self::Simple,
            "wilber" => Self::Wilber,
            "galil_park" => Self::GalilPark,
            "staggered" => Self::Staggered,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostKind {
    Sse,
    Sae,
    RoundUp,
    RoundDown,
    MaxDist,
}

impl CostKind {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "sse" => Self::Sse,
            "sae" => Self::Sae,
            "roundup" => Self::RoundUp,
            "rounddown" => Self::RoundDown,
            "maxdist" => Self::MaxDist,
            _ => return None,
        })
    }
}

fn run_dp<C: DpCost>(n: usize, k: usize, method: Method, cost: &C) -> Vec<usize> {
    match method {
        Method::Wilber => wilber(n, cost),
        Method::GalilPark => galil_park(n, cost),
        Method::Staggered => staggered(n, cost, k),
        Method::Simple => unreachable!("handled separately"),
    }
}

/// `SMALL`/`LARGE` constants of the adapted non-SSE costs.
fn range_constants(x: &[f64]) -> (f64, f64) {
    let n = (x.len() - 1) as f64;
    let small = (x[x.len() - 1] - x[0]) * n;
    (small, small * (1.0 + n))
}

fn solve_other<S: Segment>(x: &[f64], k: usize, method: Method, seg: S) -> Vec<i64> {
    let n = x.len();
    if method == Method::Simple {
        return implicit_to_explicit(&simple_concave(n, k, &seg));
    }
    let (small, large) = range_constants(x);
    let cost = Adapted::new(seg, k, small, large);
    implicit_to_explicit(&run_dp(n, k, method, &cost))
}

/// Optimal microaggregation of the **sorted** points `x`.
///
/// `stable` selects how SSE costs are evaluated: `0` fastest, `1` (default)
/// numerically stable cell-wise prefix sums, `2` plain prefix sums of `x` and
/// `x^2`; `3` (only for `Method::Simple`) recomputes every cluster directly.
/// It is ignored for other costs.
pub fn microaggregation(
    x: &[f64],
    k: usize,
    method: Method,
    cost: CostKind,
    stable: u8,
) -> Result<Vec<i64>, String> {
    let n = x.len();
    if k == 0 || k > n {
        return Err(format!("k={k} must be in 1..={n}"));
    }
    if let Some(t) = trivial_microagg(n, k) {
        return Ok(t);
    }
    // The original only short-circuits k == 1 for the simple DP; the other methods run
    // the DP, which (with ties) can merge equal points. Keep that for identical output.
    if k == 1 && method == Method::Simple {
        return Ok((0..n as i64).collect());
    }
    match cost {
        CostKind::Sse => sse(x, k, method, stable),
        CostKind::Sae => Ok(solve_other(x, k, method, SaeSegment::new(x))),
        CostKind::RoundUp => Ok(solve_other(x, k, method, RoundUpSegment::new(x))),
        CostKind::RoundDown => Ok(solve_other(x, k, method, RoundDownSegment::new(x))),
        CostKind::MaxDist => Ok(solve_other(x, k, method, MaxDistSegment { v: x })),
    }
}

fn sse(x: &[f64], k: usize, method: Method, stable: u8) -> Result<Vec<i64>, String> {
    let n = x.len();
    if method == Method::Simple {
        let r = match stable {
            0 => simple_concave(n, k, &FasterSseSegment::new(x)),
            1 => simple_concave(n, k, &StableSseSegment::new(x, 3 * k)),
            2 => simple_concave(n, k, &SseSegment::new(x)),
            3 => simple_concave(n, k, &NoPrecomputeSseSegment { v: x }),
            _ => return Err("stable must be in 0..=3".into()),
        };
        return Ok(implicit_to_explicit(&r));
    }
    let n_f = n as f64;
    let r = match stable {
        0 => {
            let seg = FasterSseMeanSegment::new(x);
            let x_bar = (seg.cumsum[n] - seg.cumsum[0]) / n_f;
            // The cluster costs `-n * mean^2` lie in `[-n * max(x^2), 0]`, so the penalty
            // constant must dominate that range (the global mean alone, as used by the
            // original, vanishes for data centered around 0 and then violates `k`).
            let max_sq = x.iter().fold(0.0f64, |m, v| m.max(v * v));
            let small = ((n_f + 1.0) * x_bar * x_bar).max(n_f * max_sq);
            run_dp(n, k, method, &Adapted::new(seg, k, small, small * (1.0 + n_f)))
        }
        1 => {
            let sum: f64 = x.iter().sum();
            let sum2: f64 = x.iter().map(|v| v * v).sum();
            let small = sse_from_sums(sum, sum2, n);
            let seg = StableSseSegment::new(x, 3 * k);
            run_dp(n, k, method, &Adapted::new(seg, k, small, small * (1.0 + n_f)))
        }
        2 => {
            let seg = SseSegment::new(x);
            let small = seg.seg(0, n);
            run_dp(n, k, method, &Adapted::new(seg, k, small, small * (1.0 + n_f)))
        }
        _ => return Err("stable must be in 0..=2 for this method".into()),
    };
    Ok(implicit_to_explicit(&r))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Exact cost of the sorted slice `v` as one cluster.
    fn seg_cost(v: &[f64], cost: CostKind) -> f64 {
        let n = v.len() as f64;
        let (first, last) = (v[0], v[v.len() - 1]);
        match cost {
            CostKind::Sse => {
                let mean = v.iter().sum::<f64>() / n;
                v.iter().map(|a| (a - mean) * (a - mean)).sum()
            }
            CostKind::Sae => {
                let med = v[v.len() / 2];
                v.iter().map(|a| (a - med).abs()).sum()
            }
            CostKind::RoundUp => v.iter().map(|a| last - a).sum(),
            CostKind::RoundDown => v.iter().map(|a| a - first).sum(),
            CostKind::MaxDist => last - first,
        }
    }

    fn labels_cost(x: &[f64], labels: &[i64], cost: CostKind) -> f64 {
        let mut total = 0.0;
        let mut start = 0;
        for i in 1..=x.len() {
            if i == x.len() || labels[i] != labels[start] {
                total += seg_cost(&x[start..i], cost);
                start = i;
            }
        }
        total
    }

    /// Optimal cost by an O(n^2) dynamic program over all cluster sizes >= k.
    fn brute_force(x: &[f64], k: usize, cost: CostKind) -> f64 {
        let n = x.len();
        let mut best = vec![f64::INFINITY; n + 1];
        best[0] = 0.0;
        for j in 1..=n {
            for i in 0..=j.saturating_sub(k) {
                if j - i >= k && best[i].is_finite() {
                    best[j] = best[j].min(best[i] + seg_cost(&x[i..j], cost));
                }
            }
        }
        best[n]
    }

    fn check_all(x: &[f64], k: usize) {
        let n = x.len();
        for cost in [CostKind::Sse, CostKind::Sae, CostKind::RoundUp, CostKind::RoundDown, CostKind::MaxDist] {
            let opt = brute_force(x, k, cost);
            for method in [Method::Simple, Method::Wilber, Method::GalilPark, Method::Staggered] {
                let stables: &[u8] = match (cost, method) {
                    (CostKind::Sse, Method::Simple) => &[0, 1, 2, 3],
                    (CostKind::Sse, _) => &[0, 1, 2],
                    _ => &[1],
                };
                for &stable in stables {
                    let labels = microaggregation(x, k, method, cost, stable).unwrap();
                    assert_eq!(labels.len(), n);
                    // contiguous, non-decreasing labels and every cluster has >= k points
                    assert!(labels.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1));
                    let mut counts = std::collections::HashMap::new();
                    for l in &labels {
                        *counts.entry(*l).or_insert(0usize) += 1;
                    }
                    assert!(counts.values().all(|&c| c >= k), "{x:?} k={k} {method:?} {cost:?}");
                    let got = labels_cost(x, &labels, cost);
                    let tol = 1e-9 * (1.0 + opt.abs());
                    assert!(
                        (got - opt).abs() <= tol,
                        "{x:?} k={k} {method:?} {cost:?} stable={stable}: cost {got} != optimal {opt}"
                    );
                }
            }
        }
    }

    #[test]
    fn all_methods_match_brute_force() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for case in 0..400 {
            let n = 1 + (rng.next() % 40) as usize;
            let mut x: Vec<f64> = (0..n)
                .map(|_| match case % 4 {
                    0 => rng.unit(),
                    1 => (rng.next() % 5) as f64, // many ties
                    2 => (rng.next() % 1000) as f64,
                    _ => 1000.0 * rng.unit().powi(3),
                })
                .collect();
            x.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let kmax = n.min(12);
            for k in 1..=kmax {
                check_all(&x, k);
            }
        }
    }

    #[test]
    fn arange_and_constant() {
        for n in 1..60usize {
            let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let c = vec![3.5; n];
            for k in [1, 2, 3, 4, 7] {
                if k <= n {
                    check_all(&x, k);
                    check_all(&c, k);
                }
            }
        }
    }

    #[test]
    fn rejects_bad_k() {
        assert!(microaggregation(&[1.0, 2.0], 0, Method::Simple, CostKind::Sse, 1).is_err());
        assert!(microaggregation(&[1.0, 2.0], 3, Method::Simple, CostKind::Sse, 1).is_err());
    }
}
