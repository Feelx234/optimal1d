//! Dynamic programs for concave one-dimensional problems
//! (least weight subsequence): Wilber (1987), Galil & Park (1989), the
//! "staggered" variant for bounded cluster sizes, and the quadratic baseline.
//!
//! All solvers return the implicit assignment `F` with `F[j]` = start of the
//! last cluster of the optimal clustering of points `0..=j`
//! (see [`crate::clustering`]).

use crate::cost::{DpCost, Segment};
use crate::smawk::{smawk, Scratch};

/// Wilber's algorithm, "The concave least weight subsequence problem revisited" (1987).
pub fn wilber<C: DpCost>(n: usize, cost: &C) -> Vec<usize> {
    wilber_impl(n, cost, false)
}

/// Wilber's algorithm with the row-bound update of the original fast1dkmeans
/// implementation: `r = c + 1` after a correction step and `r` unchanged otherwise
/// (instead of `r = F[..]`). Kept to reproduce the original cluster labels verbatim.
pub fn wilber_fast1dkmeans<C: DpCost>(n: usize, cost: &C) -> Vec<usize> {
    wilber_impl(n, cost, true)
}

fn wilber_impl<C: DpCost>(n: usize, cost: &C, orig_r: bool) -> Vec<usize> {
    let mut f = vec![0usize; n];
    let mut fv = vec![0.0f64; n + 1];
    let mut h = vec![0usize; n];
    let mut hv = vec![0.0f64; n + 1];
    let mut scratch = Scratch::new();
    let mut c = 0usize; // columns [0, c] have correct F values
    let mut r = 0usize; // rows [r, c] may contain column minima

    while c < n {
        let p = (2 * c - r + 1).min(n);
        {
            let fvr = &fv;
            // matrix entry (row j, column i) is the edge i -> j + 1
            smawk(c, p, r, c + 1, &|j, i| cost.calc(i, j + 1, fvr), &mut f, &mut scratch);
        }
        for j in c..p {
            fv[j + 1] = cost.calc(f[j], j + 1, &fv);
        }
        {
            let fvr = &fv;
            smawk(c + 1, p, c + 1, p, &|j, i| cost.calc(i, j + 1, fvr), &mut h, &mut scratch);
        }
        for j in c + 1..p {
            hv[j + 1] = cost.calc(h[j], j + 1, &fv);
        }
        let mut j0 = p + 1;
        for j in c + 2..=p {
            if hv[j] < fv[j] {
                f[j - 1] = h[j - 1];
                j0 = j;
                break;
            }
        }
        if j0 == p + 1 {
            // F values up to p (inclusive) are correct
            if !orig_r {
                r = f[p - 1];
            }
            c = p;
        } else {
            fv[j0] = hv[j0];
            r = if orig_r { c + 1 } else { f[j0 - 1] };
            c = j0;
        }
    }
    f
}

/// Galil & Park (1989), "A linear-time algorithm for concave one-dimensional dynamic programming".
pub fn galil_park<C: DpCost>(n: usize, cost: &C) -> Vec<usize> {
    let mut f = vec![0usize; n];
    let mut fv = vec![0.0f64; n + 1];
    let mut nn = vec![0usize; n];
    let mut nv = vec![f64::INFINITY; n + 1];
    let mut scratch = Scratch::new();
    let mut c = 0usize;
    let mut r = 0usize;

    while c < n {
        let p = (2 * c - r + 1).min(n);
        {
            let fvr = &fv;
            smawk(c, p, r, c + 1, &|j, i| cost.calc(i, j + 1, fvr), &mut f, &mut scratch);
        }
        for j in c..p {
            let val = cost.calc(f[j], j + 1, &fv);
            if val < nv[j + 1] {
                fv[j + 1] = val;
            } else {
                fv[j + 1] = nv[j + 1];
                f[j] = nn[j];
            }
        }
        let mut j0 = p + 2;
        for j in c + 2..=p {
            if cost.calc(j - 1, j, &fv) < fv[j] {
                // the H value considered was smaller, may not continue
                f[j - 1] = j - 1;
                j0 = j;
                fv[j0] = cost.calc(j - 1, j, &fv);
                r = c;
                c = j0;
                break;
            }
            if fv[p] <= cost.calc(j - 1, p, &fv) {
                // row j entries c..p are eliminated => continue
            } else {
                j0 = j;
                let end = (p + 1).min(n);
                nn[j - 1..end].copy_from_slice(&f[j - 1..end]);
                for t in j - 1..p {
                    nv[t] = fv[t];
                }
                r = c + 1;
                c = j0;
                break;
            }
        }
        if j0 == p + 2 {
            r = r.max(f[p - 1]);
            c = p;
        }
    }
    f
}

/// Linear time solver for bounded cluster sizes `k..=2k-1` ("staggered" blocks).
pub fn staggered<C: DpCost>(n: usize, cost: &C, k: usize) -> Vec<usize> {
    let mut f = vec![0usize; n];
    let mut fv = vec![0.0f64; n + 1];
    let mut scratch = Scratch::new();

    for i in k - 1..(2 * k - 1).min(n) {
        f[i] = 0;
        fv[i + 1] = cost.calc(0, i + 1, &fv);
    }
    if n <= 2 * k - 1 {
        return f;
    }

    let mut one_round = |f: &mut Vec<f64>, fi: &mut Vec<usize>, left: usize, r: usize, u: usize, b: usize| {
        let f_min = f[u..b].iter().cloned().fold(f64::INFINITY, f64::min);
        for x in &mut f[u..b] {
            *x -= f_min;
        }
        {
            let fvr = &*f;
            smawk(left, r, u, b, &|j, i| cost.calc(i, j + 1, fvr), fi, &mut scratch);
        }
        for j in left..r {
            f[j + 1] = cost.calc(fi[j], j + 1, f);
        }
    };

    // first block
    let max_col = (3 * k - 1).min(n);
    one_round(&mut fv, &mut f, 2 * k - 1, max_col, k, 2 * k);
    if n <= max_col {
        return f;
    }

    // remaining blocks
    let rem = n - 3 * k + 1;
    let (full, rest) = (rem / k, rem % k);
    let mut j = 3;
    for i in 3..3 + full {
        let u = f[i * k - 2].max((i - 2) * k);
        one_round(&mut fv, &mut f, i * k - 1, (i + 1) * k - 1, u, i * k);
        j = i + 1;
    }
    if rest > 0 {
        let u = f[j * k - 2].max((j - 2) * k);
        one_round(&mut fv, &mut f, j * k - 1, j * k - 1 + rest, u, (j - 1) * k + rest);
    }
    f
}

/// Quadratic time DP for microaggregation assuming the cluster cost is concave
/// (restricts the search window of the previous cluster end to be monotone).
pub fn simple_concave<S: Segment>(n: usize, k: usize, seg: &S) -> Vec<usize> {
    let mut back = vec![0usize; n]; // start of the last cluster ending at i
    let mut min_vals = vec![0.0f64; n];
    for v in min_vals.iter_mut().take(k - 1) {
        *v = f64::INFINITY;
    }
    for i in k - 1..(2 * k - 1).min(n) {
        min_vals[i] = seg.seg(0, i + 1);
        back[i] = 0;
    }
    let mut prev_min_index = 0usize;
    for right in 2 * k - 1..n {
        let mut min_index = right + 1 - 2 * k;
        let mut prev_min_val = min_vals[min_index] + seg.seg(min_index + 1, right + 1);
        for left in (right + 2 - 2 * k).max(prev_min_index)..right - k + 1 {
            let new_val = min_vals[left] + seg.seg(left + 1, right + 1);
            if new_val < prev_min_val {
                min_index = left;
                prev_min_val = new_val;
            }
        }
        back[right] = min_index + 1;
        prev_min_index = min_index;
        min_vals[right] = prev_min_val;
    }
    back
}

/// Quadratic time DP for microaggregation making no assumption on the cost.
pub fn simple_general<S: Segment>(n: usize, k: usize, seg: &S) -> Vec<usize> {
    let mut back = vec![0usize; n];
    let mut min_vals = vec![0.0f64; n];
    for v in min_vals.iter_mut().take(k - 1) {
        *v = f64::INFINITY;
    }
    for i in k - 1..(2 * k - 1).min(n) {
        min_vals[i] = seg.seg(0, i + 1);
    }
    for i in 2 * k - 1..n {
        let mut min_index = i + 1 - 2 * k;
        let mut prev = min_vals[min_index] + seg.seg(min_index + 1, i + 1);
        for j in i + 2 - 2 * k..i - k + 1 {
            let new_val = min_vals[j] + seg.seg(j + 1, i + 1);
            if new_val < prev {
                min_index = j;
                prev = new_val;
            }
        }
        back[i] = min_index + 1;
        min_vals[i] = prev;
    }
    back
}
