#pragma once

#include "linalg.h"

/**
 * @file activation.h
 * @brief Activation functions and their derivatives operating on matrices.
 *
 * All functions return newly allocated matrices; callers own and must free
 * the results. Derivatives are provided for use in backpropagation.
 */

//============================
// Activation Functions
//============================

/** @brief Sigmoid activation applied elementwise. */
Matrix* sigmoid(Matrix* m);
/** @brief Derivative of sigmoid applied elementwise. */
Matrix* sigmoid_prime(Matrix* m);

/** @brief ReLU activation applied elementwise. */
Matrix* relu(Matrix* m);
/** @brief Derivative of ReLU applied elementwise. */
Matrix* relu_prime(Matrix* m);

/** @brief Hyperbolic tangent activation applied elementwise. */
Matrix* tanh_activation(Matrix* m);
/** @brief Derivative of tanh applied elementwise. */
Matrix* tanh_prime(Matrix* m);

/** @brief Leaky ReLU activation (alpha=0.01) applied elementwise. */
Matrix* leaky_relu(Matrix* m);
/** @brief Derivative of Leaky ReLU (alpha=0.01) applied elementwise. */
Matrix* leaky_relu_prime(Matrix* m);

/** @brief Leaky ReLU with custom alpha (leak_parameter). */
Matrix* leaky_relu_with_alpha(Matrix* m, double leak_parameter);
/** @brief Derivative of Leaky ReLU with custom alpha. */
Matrix* leaky_relu_prime_with_alpha(Matrix* m, double leak_parameter);

/** @brief Sign activation: -1, 0, or +1 elementwise. */
Matrix* sign_activation(Matrix* m);
/** @brief Derivative of sign (0 everywhere; undefined at 0). */
Matrix* sign_prime(Matrix* m);

/** @brief Identity activation (returns a copy). */
Matrix* identity_activation(Matrix* m);
/** @brief Derivative of identity (ones). */
Matrix* identity_prime(Matrix* m);

/** @brief Hard Tanh activation clamped to [-1, 1]. */
Matrix* hard_tanh(Matrix* m);
/** @brief Derivative of Hard Tanh (1 in (-1,1), else 0). */
Matrix* hard_tanh_prime(Matrix* m);
