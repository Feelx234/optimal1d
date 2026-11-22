# oned-core

Rust core shared by the Python packages
[microagg1d](https://github.com/Feelx234/microagg1d) (optimal univariate microaggregation) and
[fast1dkmeans](https://github.com/Feelx234/fast1dkmeans) (optimal 1d k-means).

| module | content |
| --- | --- |
| `smawk` | iterative SMAWK row-minima search for implicit totally monotone matrices |
| `concave` | Wilber, Galil & Park, "staggered" and quadratic dynamic programs |
| `cost`, `stable` | cluster cost functions (SSE, SAE, round up/down, max distance) and DP adapters |
| `microagg` | optimal microaggregation front end |
| `kmeans` | binary search (Gronlund et al.) and matrix searching (Wu) k-means solvers |
| `clustering` | implicit/explicit cluster assignment helpers |

`cargo test --release` runs the unit tests, which compare every solver with brute-force dynamic programs.
