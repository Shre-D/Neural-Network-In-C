## Project Plan and Development Phases

### Vision

Build a small, readable, and performant neural network library with:

- A portable C core and stable C ABI
- Optional C++ RAII wrapper
- CUDA backend leveraging vendor libraries first
- Educational clarity and production-grade correctness/perf

### Guiding Principles

- Correctness first; performance second; features last
- Minimal public API; stable ABI; explicit ownership
- Determinism and strong testing
- Prefer vendor libraries for hot paths

---

## Phases Overview

### Phase 0 — Stabilization (Current)

- Fix API mismatches (`apply_onto_matrix` signature) and bias-add misuse in forward pass
- Clarify cache lifecycle (avoid double free) and naming
- Document shapes and add asserts for all linear/activation ops
- Outcome: Clean, correct single-sample MLP forward/backward

### Phase 1 — C Core + BLAS CPU Backend

- Introduce C ABI for tensors/ops/modules
- Integrate BLAS (OpenBLAS/BLIS/MKL) for GEMM
- Thread-parallel elementwise and reductions (OpenMP or pthreads)
- Outcome: CPU baseline significantly faster; stable C ABI

### Phase 2 — C++ Wrapper (Optional but Recommended)

- RAII `Tensor` wrapper, views, shape utilities; no new semantics
- Safer API with clear move semantics and no hidden syncs
- Outcome: Higher-level ergonomics with zero change to C ABI

### Phase 3 — CUDA Backend

- Device tensors, host↔device copy, pinned memory utilities
- cuBLAS for GEMM; custom kernels for elementwise/reductions; streams
- Outcome: Functional GPU backend with parity to CPU ops

### Phase 4 — Optimizers and Initializers

- SGD, Momentum, Adam; weight decay; clip gradients
- Xavier/He initializers; reproducible seeding
- Outcome: Usable training loop for small models

### Phase 5 — Optional Minimal Autograd

- Small tape-based engine for a subset of ops
- Keep explicit backprop as first-class; autograd is opt-in
- Outcome: Ergonomics without complex dynamic graph machinery

### Phase 6 — Serialization and Examples

- Save/load weights; simple binary format and versioning
- Example scripts: MLP on MNIST-like toy, small CNN (post-conv support)
- Outcome: Reproducible demos and restartable training

### Phase 7 — Testing and CI

- Golden tests for ops and gradients across backends
- Property tests for shapes/broadcast; determinism checks
- CI matrix (Linux; CPU; optional GPU runner), benchmark smoke tests
- Outcome: Regression safety and performance stability

### Phase 8 — Performance Tuning

- Op fusion (bias+activation), layout choices, pre-transposed weights
- Memory pools/arenas; workspace reuse; NUMA-aware chunking
- Mixed precision (fp16/bf16) with loss scaling
- Outcome: Competitive throughput on standard models

### Phase 9 — Docs and Developer Experience

- Deep docs for each op: math, numerics, complexity
- Debug build toggles: NaN/Inf checks, shape tracing
- Benchmark harness producing CSV; profiling guides
- Outcome: Strong educational materials and smooth dev loop

---

## Milestones and Acceptance Criteria

- M0: Phase 0 complete — All current correctness issues fixed; docs updated
- M1: Phase 1 complete — BLAS-backed CPU passes all tests; speedup ≥ 5× vs naive
- M2: Phase 3 complete — CUDA parity on supported ops; throughput ≥ CPU for large dims
- M3: Phase 4 complete — Train small MLP end-to-end to target accuracy
- M4: Phase 7 complete — CI green; baseline benchmarks tracked across commits

---

## Immediate Next Tasks (Phase 0)

- Fix `apply_onto_matrix` declaration to match implementation
- Correct bias add in `feedforward` by using returned matrix (or add inplace API)
- Resolve cache free vs clear to avoid double free; rename appropriately
- Add shape assertions and document input/weight/bias conventions

---

## Risks and Mitigations

- Divergent backends: Minimize surface; single op registry; same tests everywhere
- Performance regressions: CI benchmarks, CSV artifacts, dashboard
- Complexity creep: Keep op set minimal; resist generality until needed

---

## Tracking

We will update this plan with checkmarks per phase and link to issues/PRs as deliverables land.
