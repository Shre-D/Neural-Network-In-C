# Neural Network in C

A from-scratch implementation of a simple feedforward neural network, written entirely in C. This project was built to gain a deep, first-principles understanding of the algorithms that power modern deep learning, focusing on implementation of the basics from scratch.

## Project Goals

The goal of this project is not to compete with established libraries like PyTorch or TensorFlow, but to deconstruct them. By implementing core components from matrix multiplication to backpropagation in a low-level language like C, this work demonstrates the mathematical concepts and the engineering challenges involved in building deep learning systems from the ground up.

### Key Features

* **First-Principles Implementation:** Successfully implemented feedforward, backpropagation, and gradient descent without external ML dependencies.
* **Custom Linear Algebra Engine:** Developed and optimized essential linear algebra operations from scratch, with a focus on explicit memory management.
* **Good C Practices:** Built as a clean, well-documented, and robust library, incorporating best practices for C development and defensive programming.
* **Comprehensive Testing:** Established a solid testing framework using CUnit and a Continuous Integration (CI) pipeline to ensure correctness and prevent regressions.

---

## Getting Started

### Prerequisites

To build and run this project, you will need the following tools:

* A C compiler (`gcc` or `clang`)
* `make`
* `pkg-config`
* **CUnit** : A unit testing framework for C.
* On Debian/Ubuntu: `sudo apt-get install libcunit1-dev`
* On macOS (with Homebrew): `brew install cunit`
* `clang-format` (for code formatting)

### Installation & Setup

1. **Clone the repository:**
   **Bash**

   ```
   git clone https://github.com/Shre-D/Neural-Network-In-C.git
   cd Neural-Network-In-C
   ```
2. Set up Git Hooks:
   The project includes pre-commit hooks for formatting. Make them executable by running the initialization script:
   **Bash**

   ```
   ./init.sh
   ```

### Building the Project

The provided `Makefile` automates the build process for the library, tests, and examples.

* **Build everything (library, tests, and examples):**
  **Bash**

  ```
  make all
  ```
* Run the test suite:
  This will compile the library and tests, then execute the test runner.
  **Bash**

  ```
  make test
  ```
* Run example programs:
  The example executables are built in the nn/src/examples/ directory.
  **Bash**

  ```
  # After running 'make all'
  ./nn/src/examples/your_example_name
  ```
* **Clean build artifacts:**
  **Bash**

  ```
  make clean
  ```
* **Format code:**
  **Bash**

  ```
  make format
  ```
* **Generate docs:**
  **Bash**

  ```
  cd docs && doxygen Doxyfile
  ```

---

## Architecture and Core Concepts

The library is designed with clear modular boundaries to separate concerns.

### Project Structure

```
.
├── .github/workflows/      # Continuous Integration (CI) pipeline
├── nn/
│   ├── include/            # Public header files for the library
│   └── src/                # C source file implementations
│       ├── activation/
│       ├── cache/
|	    ├── examples/       # Examples, featuring XOR and MNIST
│       ├── linalg/
│       ├── loss/
│       └── neural_network/
│       └── utils/        
├── scripts/                # Git hooks and initialization scripts
├── tests/                  # CUnit test suites
└── Makefile                # Build automation
```

### 1. Matrix (`linalg`)

The `Matrix` struct is a row/column typed wrapper over a contiguous `double*` buffer. The API provides creation, copying, transpose, dot product, element-wise operations, and scaling. Memory ownership is explicit; callers are responsible for freeing returned matrices using `free_matrix`. 1D matrices were chosen as the base structure because of higher performance in matrix calculcations than 2D matrices, due to memroy localization being better in the former.

### 2. Activations (`activation`)

Activation functions are implemented as matrix-to-matrix functions with the signature Matrix* activation(Matrix* m). Their corresponding derivatives share the same signature.

Supported: Sigmoid, ReLU, Tanh, Leaky ReLU, Sign, Identity, and HardTanh.

### 3. Loss Functions (`loss`)

Loss functions operate on predictions (y_hat) and ground truth (y). The API provides both scalar loss values and matrix gradients.

Supported: Mean Squared Error (MSE), Cross-Entropy (CCE), Mean Absolute Error (MAE), and Binary Cross-Entropy (BCE).

### 4. Neural Network & Data Flow

The `NeuralNetwork` orchestrates the forward and backward passes. It contains an array of `Layer` pointers, where each `Layer` holds its `weights`, `bias`, and an activation function pointer.

#### Data Flow and Shapes (Single Sample)

* Input **X**: shape **(**1**×**D**_**in**)**
* For a layer **i**:
  * Previous activation **a**_**i**−**1**: shape **(**1**×**D**_**in**)**
  * Weights **W**_**i**: shape **(**D**_**in**×**D**_**o**u**t**)**
  * Bias **b**_**i**: shape **(**1**×**D**_**o**u**t**)**

**Forward Pass:**

1. Compute pre-activation: **z**_**i**=**a**_**i**−**1**⋅**W**_**i**+**b**_**i**
2. Compute activation: **a**_**i**=**activation**(**z**_**i**)

**Backward Pass:**

1. Compute initial error at the last layer (**L**): **δ**_**L**=**∂**a**_**L**∂**L****⊙**a**′**(**z**_**L**)**, where **⊙** is the element-wise product.
2. Propagate error to previous layers: **δ**_**i**=**(**δ**_**i**+**1**⋅**W**_**i**+**1**T**)**⊙**a**′**(**z**_**i**)
3. Compute gradients:
   * Weight gradient: **∇_**W**_**i**L**=**a**_**i**−**1**T**⋅**δ**_**i
   * Bias gradient: **∇_**b**_**i**L**=**δ**_**i** (summed across a batch)

An internal key-value `cache` stores intermediate matrices (`input`, `z_i`, `a_i`, `delta_i`) required for backpropagation.

### Usage Sketch

**C**

```
#include "neural_network.h"

// 1. Define network architecture and initialize weights
NeuralNetwork* nn = create_network(num_layers);
// ... setup layers, weights, biases, and activation functions ...

// 2. Forward pass to get a prediction
Matrix* y_hat = feedforward(nn, input_data);

// 3. Backward pass to compute gradients
backpropagate(nn, y_true, mean_squared_error, mean_squared_error_gradient);
Matrix* dW1 = calculate_weight_gradient(nn->cache, /*layer_index=*/1, ...);
Matrix* db1 = calculate_bias_gradient(nn->cache, /*layer_index=*/1, ...);

// 4. Update parameters (optimizer logic)
// W1 = W1 - learning_rate * dW1;

// 5. Cleanup
free_matrix(y_hat);
free_matrix(dW1);
free_matrix(db1);
free_network(nn);
```

---

## Roadmap & Future Development

This C implementation serves as a robust and well-understood foundation. The next phase will focus on pushing the boundaries of performance and capability.

* [ ] **High-Performance Re-implementation:** Port the library to C++ and CUDA to leverage GPU acceleration.
* [ ] **BLAS Integration:** Integrate optimized libraries like OpenBLAS for matrix operations via a compile-time flag.
* [ ] **Optimizer Suite:** Implement a suite of standard optimizers (`SGD`, `Momentum`, `Adam`).
* [ ] **Batch Processing:** Add support for mini-batch training to improve gradient stability and training speed.
* [ ] **Serialization:** Develop utilities for saving and loading trained network weights to/from disk.
* [ ] **Website:** Website to show the features, training process and a few results obtained. 
* [X] ~~Establish a solid C foundation with comprehensive testing.~~ (Completed)

## Known Issues & Planned Improvements

This project was a learning experience, and as such, there are several areas identified for improvement in future iterations:

1. **Memory Efficiency:** Introduce in-place variants of matrix operations (e.g., `add_inplace`) to reduce allocation overhead in performance-critical loops.
2. **Cache Performance:** The cache currently performs deep copies on puts/gets. A non-copy `peek` or a reference-counting mechanism would significantly reduce overhead.
3. **API Design:** Refine function signatures for better `const`-correctness and consistency. For example, storing the `activation_prime` function pointer directly in the `Layer` struct.
4. **Logging:** Reduce default logging verbosity and implement a runtime-configurable log level.
5. **Model Summaries and Logs:** The Model Summaries and Logs are incomplete as of now. I'm working on implementing these.
6. **MNIST:** This example has to be evaluated end to end and reviewed, since I've encountered a few bugs.
