//! Helpers for the implicit cluster representation used by the DP solvers.
//!
//! An implicit result `r` has `r[j]` = start index of the cluster that ends
//! at (and includes) point `j`.

/// Converts an implicit assignment into explicit 0-based cluster labels.
pub fn implicit_to_explicit(result: &[usize]) -> Vec<i64> {
    let n = result.len();
    let num_clusters = num_clusters(result);
    let mut out = vec![0i64; n];
    let mut label = num_clusters as i64 - 1;
    let mut pos = n - 1;
    while result[pos] > 0 {
        for o in &mut out[result[pos]..=pos] {
            *o = label;
        }
        pos = result[pos] - 1;
        label -= 1;
    }
    for o in &mut out[..=pos] {
        *o = label;
    }
    out
}

/// Number of clusters encoded in an implicit assignment.
pub fn num_clusters(result: &[usize]) -> usize {
    let mut count = 1;
    let mut pos = result.len() - 1;
    while result[pos] > 0 {
        pos = result[pos] - 1;
        count += 1;
    }
    count
}

/// Splits sorted points into the trivial cases of microaggregation
/// (`2k > n` gives one cluster, `2k == n` gives two).
pub fn trivial_microagg(n: usize, k: usize) -> Option<Vec<i64>> {
    if 2 * k > n {
        Some(vec![0; n])
    } else if 2 * k == n {
        Some((0..n).map(|i| (i >= k) as i64).collect())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit() {
        // clusters [0,1) [1,2) [2,6)
        assert_eq!(implicit_to_explicit(&[0, 1, 2, 2, 2, 2]), vec![0, 1, 2, 2, 2, 2]);
        assert_eq!(implicit_to_explicit(&[0, 0, 0, 3, 3, 3]), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(num_clusters(&[0, 0, 0, 3, 3, 3]), 2);
    }
}
