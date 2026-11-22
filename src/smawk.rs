//! Iterative SMAWK row-minima search for totally monotone matrices.
//!
//! The matrix is given implicitly by a closure `f(row, col)`. Rows are the
//! integers in `row_start..row_stop`, columns the integers in
//! `col_start..col_stop`. For each row the (leftmost) column of its minimum is
//! written to `result[row]`.
//!
//! The implementation follows
//! <https://courses.engr.illinois.edu/cs473/sp2016/notes/06-sparsedynprog.pdf>
//! but is iterative (no recursion) and re-uses its scratch space across calls.

/// Reusable scratch space for [`smawk`].
#[derive(Default)]
pub struct Scratch {
    buf: Vec<usize>,
    /// `ranges[d]` is the range in `buf` of the columns that survive the
    /// reduction at depth `d` (i.e. the input columns of depth `d + 1`).
    ranges: Vec<(usize, usize)>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Computes the row minima of the implicit matrix `f` (see module docs).
///
/// Ties are broken towards the smaller column.
pub fn smawk<F: Fn(usize, usize) -> f64>(
    row_start: usize,
    row_stop: usize,
    col_start: usize,
    col_stop: usize,
    f: &F,
    result: &mut [usize],
    scratch: &mut Scratch,
) {
    if row_stop <= row_start || col_stop <= col_start {
        return;
    }
    let m = row_stop - row_start;
    let buf = &mut scratch.buf;
    let ranges = &mut scratch.ranges;
    buf.clear();
    buf.extend(col_start..col_stop);
    ranges.clear();
    // `cur` is the range of the columns used at the current depth
    let mut cur = (0usize, col_stop - col_start);

    // ---- reduce phase, rows 2^d - 1, 2*2^d - 1, ... (relative to row_start)
    let mut depth = 0usize;
    // set if the recursion stopped because a single column is left
    let mut single_col_depth: Option<usize> = None;
    loop {
        let step = 1usize << depth;
        let n_rows = m / step;
        if n_rows == 0 {
            break;
        }
        let first_row = row_start + step - 1;
        if cur.1 - cur.0 == 1 {
            let c = buf[cur.0];
            let mut r = first_row;
            while r < row_stop {
                result[r] = c;
                r += step;
            }
            single_col_depth = Some(depth);
            break;
        }
        // reduce: keep at most n_rows candidate columns
        let out_start = cur.1;
        buf.resize(out_start + n_rows, 0);
        buf[out_start] = buf[cur.0];
        let mut r: isize = 0;
        for idx in cur.0..cur.1 {
            let col = buf[idx];
            if col != buf[out_start + r as usize] {
                while r >= 0 {
                    let row = first_row + step * r as usize;
                    if f(row, buf[out_start + r as usize]) > f(row, col) {
                        r -= 1;
                    } else {
                        break;
                    }
                }
            }
            if (r as usize).wrapping_add(1) < n_rows {
                r += 1;
                buf[out_start + r as usize] = col;
            }
        }
        let kept = (r + 1) as usize;
        let next = (out_start, out_start + kept);
        ranges.push(next);
        result[first_row] = buf[next.0];
        cur = next;
        depth += 1;
    }

    // ---- interpolate phase: fill in the remaining rows from coarse to fine
    let deepest = match single_col_depth {
        Some(d) => d, // rows at depth d are already assigned
        None => depth,
    };
    let mut d = deepest;
    loop {
        // skip the depth where a single column finished everything
        let skip = single_col_depth == Some(d);
        let step = 1usize << d;
        let n_rows = m / step;
        if n_rows > 2 && !skip && d < ranges.len() {
            let (cs, ce) = ranges[d];
            interpolate(
                row_start + step - 1,
                step,
                n_rows,
                &buf[cs..ce],
                f,
                result,
            );
        }
        if d == 0 {
            break;
        }
        d -= 1;
    }
}

/// Fills in the rows with even index (>= 2) at one recursion level.
#[inline]
fn interpolate<F: Fn(usize, usize) -> f64>(
    first_row: usize,
    step: usize,
    n_rows: usize,
    cols: &[usize],
    f: &F,
    result: &mut [usize],
) {
    let mut cur = 0usize;
    let mut i = 2;
    while i < n_rows {
        let start_col = result[first_row + step * (i - 1)];
        let stop_col = if i + 1 < n_rows {
            result[first_row + step * (i + 1)]
        } else {
            cols[cols.len() - 1]
        };
        while cols[cur] < start_col {
            cur += 1;
        }
        let row = first_row + step * i;
        let mut best = cur;
        let mut best_val = f(row, cols[best]);
        while cols[cur] < stop_col {
            let tmp = f(row, cols[cur + 1]);
            if best_val > tmp {
                best = cur + 1;
                best_val = tmp;
            }
            cur += 1;
        }
        result[row] = cols[best];
        i += 2;
    }
}

/// Convenience: row minima of a dense row-major matrix (used for testing).
pub fn smawk_dense(a: &[f64], n_rows: usize, n_cols: usize) -> Vec<usize> {
    let mut out = vec![0usize; n_rows];
    let mut s = Scratch::new();
    smawk(0, n_rows, 0, n_cols, &|r, c| a[r * n_cols + c], &mut out, &mut s);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// Random totally monotone matrix: a_ij = (x_i - y_j)^2 with sorted x, y is Monge.
    fn monge(rng: &mut Lcg, m: usize, n: usize) -> Vec<f64> {
        let mut xs: Vec<f64> = (0..m).map(|_| (rng.next() * 8.0).floor()).collect();
        let mut ys: Vec<f64> = (0..n).map(|_| (rng.next() * 8.0).floor()).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut a = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                a[i * n + j] = (xs[i] - ys[j]).powi(2);
            }
        }
        a
    }

    #[test]
    fn matches_bruteforce() {
        let mut rng = Lcg(42);
        for m in 1..40 {
            for n in 1..40 {
                let a = monge(&mut rng, m, n);
                let got = smawk_dense(&a, m, n);
                for i in 0..m {
                    let row = &a[i * n..(i + 1) * n];
                    let min = row.iter().cloned().fold(f64::INFINITY, f64::min);
                    // leftmost minimum
                    let want = row.iter().position(|&v| v == min).unwrap();
                    assert_eq!(row[got[i]], min, "m={m} n={n} row={i}");
                    assert_eq!(got[i], want, "tie-break m={m} n={n} row={i}");
                }
            }
        }
    }
}
