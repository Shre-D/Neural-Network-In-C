## Neural Network in C — Project Documentation

### Overview
This project implements a simple feedforward neural network in C with clear modular boundaries:
- Matrix operations (`linalg`)
- Activation functions and their derivatives (`activation`)
- Loss functions and their gradients (`loss`)
- A lightweight key-value cache for forward/backward intermediates (`cache`)
- Network orchestration and passes (`neural_network`, `feedforward`, `backprop`)

The code emphasizes explicit memory management, defensive checks via assertions, and readable logging macros.

### Directory Structure
- `nn/include/`: Public headers for the library modules
- `nn/src/`: Module implementations
  - `activation/`: Activation functions and derivatives
  - `cache/`: Internal cache implementation
  - `linalg/`: Matrix IO and operations
  - `loss/`: Loss functions and gradients
  - `neural_network/`: Forward and backward passes
  - `main.c`: Example entry point (if used)
- `scripts/`: Init and pre-commit hooks
- `build/`: Build artifacts (ignored)

### Core Concepts
#### Matrix (`linalg`)
`Matrix` is a row/column typed wrapper over a contiguous `double*` buffer. The API provides creation, copying, printing, transpose, dot, elementwise ops, scaling, and simple IO. Functions typically allocate new matrices as results (out-of-place). Memory ownership is explicit and callers must `free_matrix` returned results.

#### Activations (`activation`)
Activations are implemented as matrix-to-matrix functions: `Matrix* activation(Matrix* m)`. Derivatives are provided with the same signature. Supported: Sigmoid, ReLU, Tanh, Leaky ReLU (with optional alpha), Sign, Identity, HardTanh.

#### Losses (`loss`)
Loss functions operate on `y_hat` and `y` with both scalar loss and gradient forms. Supported: MSE, CCE, MAE, BCE. Gradients return a `Matrix*` matching the shape of `y_hat`.

#### Cache (`cache`)
An internal hashmap stores deep copies of matrices by string keys to pass intermediates between forward and backward passes. Keys like `input`, `z_i`, `a_i`, `delta_i` are used. Retrieval returns deep copies to avoid in-place corruption.

#### Neural Network (`neural_network`, `feedforward`, `backprop`)
- `NeuralNetwork` contains an array of `Layer` pointers. Each `Layer` has `weights`, `bias`, and an `activation` function pointer. For Leaky ReLU, a per-layer `leak_parameter` is supported.
- `feedforward` computes successive `z_i = a_{i-1} · W_i + b_i`, applies activation to get `a_i`, and caches `z_i` and `a_i`. The input is cached as `input` and the final activation is the prediction `y_hat`.
- `backpropagate` takes `y_true`, a loss function and its gradient, computes `delta_L = dL/da .* a'(z_L)`, stores `delta_i` for all layers via standard backprop recursion, and exposes helpers to compute gradients for weights and biases:
  - `grad_W_i = a_{i-1}^T · delta_i`
  - `grad_b_i = delta_i` (for single-sample)

### Data Flow and Shapes
Let the following hold for a single sample (no batching):
- Input `X` shape matches the left multiplicand of the first layer dot product.
- For a layer `i`: `a_{i-1} (1×D_in)`, `W_i (D_in×D_out)`, `b_i (1×D_out)`
- Compute `z_i = a_{i-1} · W_i + b_i` yields `(1×D_out)`; then `a_i = activation(z_i)`
- For backprop: `delta_i` has shape `(1×D_out)`
- Weight gradients: `a_{i-1}^T (D_in×1)` dot `delta_i (1×D_out)` yields `(D_in×D_out)`
- Bias gradients equal `delta_i` for single-sample; for batches, bias gradients are the sum across samples.

Keys used in cache:
- `input`: copy of initial input matrix
- `z_i`: pre-activation for layer i
- `a_i`: activation for layer i
- `delta_i`: backpropagated error for layer i

### Error Handling and Logging
`ASSERT` ensures preconditions (non-null pointers, compatible shapes). Logging macros (`LOG_INFO`, etc.) emit operation-level messages. Minimum log level is set via `MIN_LOG_LEVEL` in `utils.h`.

### Usage Sketch
1) Build your network
```c
NeuralNetwork* nn = create_network(num_layers);
// allocate and assign Layer* entries, set weights, biases, activation funcs
```

2) Forward pass
```c
Matrix* y_hat = feedforward(nn, input);
```

3) Backward pass and gradients
```c
backpropagate(nn, y_true, mean_squared_error, mean_squared_error_gradient);
Matrix* dW = calculate_weight_gradient(nn->cache, layer_index, nn->num_layers);
Matrix* db = calculate_bias_gradient(nn->cache, layer_index, nn->num_layers);
```

4) Update parameters (optimizer not shown)
```c
// W_i = W_i - lr * dW;  b_i = b_i - lr * db
```

5) Cleanup
```c
free_matrix(y_hat);
free_matrix(dW);
free_matrix(db);
free_network(nn);
```

### Build and Run
- Use your preferred C toolchain (e.g., `gcc`/`clang`).
- Ensure `nn/include` is in include paths and `nn/src/**` compiled and linked.
- Pre-commit hooks in `scripts/pre-commit-hooks` can help format and lint.

### Improvements and Refactors
1) API and Correctness
- Fix `apply_onto_matrix` signature mismatch in header vs implementation.
- In `feedforward`, use the return value of `add_matrix` (or provide in-place add) to ensure bias is applied and avoid leaks.
- Avoid passing `Matrix* (*)(Matrix*)` activations into `apply_onto_matrix(double (*)(double), ...)`. Either:
  - Call activation functions directly for matrices, or
  - Redesign activations to scalar functions and keep `apply_onto_matrix`.

2) Shapes and Documentation
- Document shape conventions clearly in this file and assert them centrally in forward/backward code paths.
- Consider a helper that validates `a_{i-1} (1×D_in)`, `W (D_in×D_out)`, `b (1×D_out)` before computing.

3) Memory and Performance
- Introduce in-place variants: `add_inplace`, `mul_inplace`, `scale_inplace` to reduce allocations.
- Cache API currently deep-copies on put/get; add a non-copy `peek_matrix` for internal use or a reference-counting mechanism to cut copies.
- Add simple buffer reuse for temporary matrices in hot paths.

4) Activation Derivatives
- Store `activation_prime` in `Layer` alongside `activation` (and parameters like `leak_parameter`) to avoid brittle function-pointer comparisons.

5) Cache Lifecycle Naming
- Clarify cache freeing: either rename `clear_cache` to `free_cache` (and free the struct) or split into `clear_cache_entries` and `free_cache` to avoid double-free hazards. Ensure `free_network` calls the correct one.

6) Const-correctness
- Mark read-only arguments as `const` throughout headers (e.g., activations, derivatives when not modifying inputs) for safer APIs.

7) Logging
- Reduce verbosity of matrix operation logs to `DEBUG` to keep normal runs clean. Consider a runtime-configurable log level.

8) Testing and Examples
- Add unit tests for linalg ops, activation/derivatives parity checks, loss/gradient consistency, and forward/backward shape checks.
- Provide a minimal training example demonstrating a full loop with a tiny dataset and a simple optimizer.

9) Optimizers
- Implement basic optimizers (`SGD`, `Momentum`, `Adam`) in the `optimisers` module with a simple interface operating on `Layer` params.

10) Future Directions
- Batch support: define batch-major shapes and reduction semantics; update cache keys to include batch dimension if needed.
- Optional BLAS integration for `dot_matrix` with compile-time flag.
- Serialization utilities for saving/loading network weights.


