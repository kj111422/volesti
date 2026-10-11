# High-dimensional convex-body inference (research harness)

Follow-up to the geometric-inference prototype: distinguishing a Euclidean
ball from a Birkhoff polytope and a random inscribed polytope in dimensions
above 20, with observations no more than the dimension. See the top-of-file
comments in each target for the exact protocol, seeds, and gate criteria.

## Targets

- `reference_tests` — mathematical reference checks independent of the
  samplers (cube/ball support and covariance, V-polytope support oracle,
  Birkhoff assignment oracle and coordinate mapping, independent LP
  membership, seeded statistical checks). Fast (~1 s).

## Build

```sh
mkdir -p build && cd build
cmake .. && make -j4
./reference_tests
```

`VOLESTI_DIR` defaults to this checkout; override with
`-DVOLESTI_DIR=<path>` when building elsewhere. First configure downloads
Eigen and lp_solve into `external/_deps` (about 1 GB, git-ignored).

## Notes

- Support-function gaps are ground-truth oracles (known body required),
  never classifier features.
- Upstream `develop` diagnostic and benchmark infrastructure is reused,
  not duplicated here.
