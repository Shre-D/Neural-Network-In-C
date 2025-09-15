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

/**
 * @brief Sigmoid activation applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with sigmoid applied.
 */
Matrix* sigmoid(Matrix* m);
/**
 * @brief Derivative of sigmoid applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with sigmoid derivative applied.
 */
Matrix* sigmoid_prime(Matrix* m);

/**
 * @brief ReLU activation applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with ReLU applied.
 */
Matrix* relu(Matrix* m);
/**
 * @brief Derivative of ReLU applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with ReLU derivative applied.
 */
Matrix* relu_prime(Matrix* m);

/**
 * @brief Hyperbolic tangent activation applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with tanh applied.
 */
Matrix* tanh_activation(Matrix* m);
/**
 * @brief Derivative of tanh applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with tanh derivative applied.
 */
Matrix* tanh_prime(Matrix* m);

/**
 * @brief Leaky ReLU activation (alpha=0.01) applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with Leaky ReLU applied.
 */
Matrix* leaky_relu(Matrix* m);
/**
 * @brief Derivative of Leaky ReLU (alpha=0.01) applied elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with Leaky ReLU derivative applied.
 */
Matrix* leaky_relu_prime(Matrix* m);

/**
 * @brief Leaky ReLU with custom alpha (leak_parameter).
 * @param m Input matrix (m x n).
 * @param leak_parameter The alpha value for the leak.
 * @return New matrix (m x n) with Leaky ReLU applied.
 */
Matrix* leaky_relu_with_alpha(Matrix* m, double leak_parameter);
/**
 * @brief Derivative of Leaky ReLU with custom alpha.
 * @param m Input matrix (m x n).
 * @param leak_parameter The alpha value for the leak.
 * @return New matrix (m x n) with Leaky ReLU derivative applied.
 */
Matrix* leaky_relu_prime_with_alpha(Matrix* m, double leak_parameter);

/**
 * @brief Sign activation: -1, 0, or +1 elementwise.
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with sign activation applied.
 */
Matrix* sign_activation(Matrix* m);
/**
 * @brief Derivative of sign (0 everywhere; undefined at 0).
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) of zeros.
 */
Matrix* sign_prime(Matrix* m);

/**
 * @brief Identity activation (returns a copy).
 * @param m Input matrix (m x n).
 * @return New matrix (m x n), a copy of the input.
 */
Matrix* identity_activation(Matrix* m);
/**
 * @brief Derivative of identity (ones).
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) of ones.
 */
Matrix* identity_prime(Matrix* m);

/**
 * @brief Hard Tanh activation clamped to [-1, 1].
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with hard tanh applied.
 */
Matrix* hard_tanh(Matrix* m);
/**
 * @brief Derivative of Hard Tanh (1 in (-1,1), else 0).
 * @param m Input matrix (m x n).
 * @return New matrix (m x n) with hard tanh derivative applied.
 */
Matrix* hard_tanh_prime(Matrix* m);
