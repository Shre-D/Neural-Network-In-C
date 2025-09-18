/**
 * @file activation.c
 * @brief Implementations of activation functions and their derivatives.
 *
 * Elementwise activations used in forward passes and their corresponding
 * derivatives used in backpropagation. All functions allocate new matrices;
 * callers are responsible for freeing returned results.
 */
#include "activation.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"
#include "utils.h"

//============================
// Sigmoid Activation
//============================

/**
 * @brief Applies the sigmoid activation function element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the sigmoid function applied to each element.
 */
Matrix* sigmoid(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying sigmoid activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = 1.0 / (1.0 + exp(-m->matrix_data[i]));
  }
  return result;
}

/**
 * @brief Computes the derivative of the sigmoid activation function
 * element-wise to a matrix.
 * @param m The input matrix (output of the sigmoid function).
 * @return A new matrix with the sigmoid derivative applied to each element.
 */
Matrix* sigmoid_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying sigmoid_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    double sigmoid_val = 1.0 / (1.0 + exp(-m->matrix_data[i]));
    result->matrix_data[i] = sigmoid_val * (1.0 - sigmoid_val);
  }
  return result;
}

//============================
// ReLU Activation
//============================

/**
 * @brief Applies the ReLU (Rectified Linear Unit) activation function
 * element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the ReLU function applied to each element.
 */
Matrix* relu(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying ReLU activation to a %zux%zu matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = m->matrix_data[i];
    } else {
      result->matrix_data[i] = 0;
    }
  }
  return result;
}

/**
 * @brief Computes the derivative of the ReLU activation function element-wise
 * to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the ReLU derivative applied to each element.
 */
Matrix* relu_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying ReLU_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = 1;
    } else {
      result->matrix_data[i] = 0;
    }
  }
  return result;
}

//============================
// Tanh Activation
//============================

/**
 * @brief Applies the Hyperbolic Tangent (tanh) activation function element-wise
 * to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the tanh function applied to each element.
 */
Matrix* tanh_activation(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Tanh activation to a %zux%zu matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = tanh(m->matrix_data[i]);
  }
  return result;
}

/**
 * @brief Computes the derivative of the Hyperbolic Tangent (tanh) activation
 * function element-wise to a matrix.
 * @param m The input matrix (output of the tanh function).
 * @return A new matrix with the tanh derivative applied to each element.
 */
Matrix* tanh_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Tanh_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    double tanh_val = tanh(m->matrix_data[i]);
    result->matrix_data[i] = 1.0 - pow(tanh_val, 2);
  }
  return result;
}

//============================
// Leaky ReLU Activation
//============================

/**
 * @brief Applies the Leaky ReLU activation function element-wise to a matrix.
 * @param m The input matrix.
 * @param leak_parameter The leak parameter (alpha) for the Leaky ReLU. Must be
 * non-negative.
 * @return A new matrix with the Leaky ReLU function applied to each element.
 */
Matrix* leaky_relu(Matrix* m, double leak_parameter) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  // If I converted a non acceptable value of alpha into 0.01, it would bring
  // in debug troubles.
  ASSERT(leak_parameter >= 0.0, "Alpha value must be non-negative.");

  LOG_INFO(
      "Applying Leaky ReLU with leak_parameter=%.2f activation function to a "
      "%zux%zu "
      "matrix.",
      leak_parameter, m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = m->matrix_data[i];
    } else {
      result->matrix_data[i] = leak_parameter * m->matrix_data[i];
    }
  }

  return result;
}

/**
 * @brief Computes the derivative of the Leaky ReLU activation function
 * element-wise to a matrix.
 * @param m The input matrix.
 * @param leak_parameter The leak parameter (alpha) for the Leaky ReLU. Must be
 * non-negative.
 * @return A new matrix with the Leaky ReLU derivative applied to each element.
 */
Matrix* leaky_relu_prime(Matrix* m, double leak_parameter) {
  ASSERT(m != NULL, "Input matrix for leaky_relu_prime is NULL.");
  ASSERT(leak_parameter >= 0.0, "Alpha value must be non-negative.");
  LOG_INFO(
      "Applying Leaky ReLU with alpha=%.2f derivative to a %zux%zu matrix.",
      leak_parameter, m->rows, m->cols);
  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;

  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = 1.0;
    } else {
      result->matrix_data[i] = leak_parameter;
    }
  }

  return result;
}

//============================
// Sign Activation
//============================

/**
 * @brief Applies the Sign activation function element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the Sign function applied to each element.
 */
Matrix* sign_activation(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Sign activation to a %zux%zu matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = 1.0;
    } else if (m->matrix_data[i] < 0) {
      result->matrix_data[i] = -1.0;
    } else {
      result->matrix_data[i] = 0.0;
    }
  }
  return result;
}

/**
 * @brief Computes the derivative of the Sign activation function element-wise
 * to a matrix. The derivative of the sign function is 0 everywhere except at 0,
 * where it is undefined. For backpropagation, the derivative is commonly
 * approximated as 0.
 * @param m The input matrix.
 * @return A new matrix with the Sign derivative applied to each element (all
 * zeros).
 */
Matrix* sign_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Sign_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);
  // The derivative of the sign function is 0 everywhere except at 0, where it
  // is undefined. For backpropagation, the derivative is commonly
  // approximated as 0.
  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = 0.0;
  }
  return result;
}

//============================
// Identity Activation
//============================

/**
 * @brief Applies the Identity activation function element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix that is a copy of the input matrix.
 */
Matrix* identity_activation(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Identity activation to a %zux%zu matrix.", m->rows,
           m->cols);
  Matrix* result = copy_matrix(m);
  ASSERT(result != NULL, "Failed to copy matrix.");
  return result;
}

/**
 * @brief Computes the derivative of the Identity activation function
 * element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix of the same size as the input, with all elements set
 * to 1.0.
 */
Matrix* identity_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Identity_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = 1.0;
  }
  return result;
}

//============================
// Hard Tanh Activation
//============================

/**
 * @brief Applies the Hard Tanh activation function element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the Hard Tanh function applied to each element.
 */
Matrix* hard_tanh(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Hard Tanh activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 1.0) {
      result->matrix_data[i] = 1.0;
    } else if (m->matrix_data[i] < -1.0) {
      result->matrix_data[i] = -1.0;
    } else {
      result->matrix_data[i] = m->matrix_data[i];
    }
  }
  return result;
}

/**
 * @brief Computes the derivative of the Hard Tanh activation function
 * element-wise to a matrix.
 * @param m The input matrix.
 * @return A new matrix with the Hard Tanh derivative applied to each element.
 */
Matrix* hard_tanh_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Hard Tanh_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > -1.0 && m->matrix_data[i] < 1.0) {
      result->matrix_data[i] = 1.0;
    } else {
      result->matrix_data[i] = 0.0;
    }
  }
  return result;
}

//============================
// Softmax Activation
//============================

/**
 * @brief Applies the Softmax activation function to a matrix.
 * This function is typically used in the output layer of a neural network for
 * multi-class classification. It normalizes the input values into a probability
 * distribution.
 * @param m The input matrix.
 * @return A new matrix with the Softmax function applied.
 */
Matrix* softmax(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying softmax activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");

  for (size_t i = 0; i < m->rows; i++) {
    double max_val = m->matrix_data[i * m->cols];
    for (size_t j = 1; j < m->cols; j++) {
      if (m->matrix_data[i * m->cols + j] > max_val) {
        max_val = m->matrix_data[i * m->cols + j];
      }
    }

    double sum = 0.0;
    for (size_t j = 0; j < m->cols; j++) {
      sum += exp(m->matrix_data[i * m->cols + j] - max_val);
    }

    for (size_t j = 0; j < m->cols; j++) {
      result->matrix_data[i * m->cols + j] =
          exp(m->matrix_data[i * m->cols + j] - max_val) / sum;
    }
  }
  return result;
}

/**
 * @brief Computes the derivative of the Softmax activation function
 * element-wise to a matrix. This is typically used in conjunction with a loss
 * function like cross-entropy where the derivative simplifies to output * (1 -
 * output).
 * @param m The input matrix (output of the softmax function).
 * @return A new matrix with the Softmax derivative applied to each element.
 */
Matrix* softmax_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying softmax_prime activation to a %zux%zu matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix.");
  size_t total_elements = m->rows * m->cols;
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = m->matrix_data[i] * (1.0 - m->matrix_data[i]);
  }
  return result;
}

const char* activation_to_string(activation_function func) {
  switch (func) {
    case SIGMOID:
      return "SIGMOID";
    case RELU:
      return "RELU";
    case LEAKY_RELU:
      return "LEAKY_RELU";
    case SOFTMAX:
      return "SOFTMAX";
    default:
      return "UNKNOWN";
  }
}
