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

Matrix* sigmoid(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying sigmoid activation to a %dx%d matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    result->matrix_data[i] = 1.0 / (1.0 + exp(-m->matrix_data[i]));
  }
  return result;
}

Matrix* sigmoid_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying sigmoid_prime activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    double sigmoid_val = 1.0 / (1.0 + exp(-m->matrix_data[i]));
    result->matrix_data[i] = sigmoid_val * (1.0 - sigmoid_val);
  }
  return result;
}

//============================
// ReLU Activation
//============================

Matrix* relu(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying ReLU activation to a %dx%d matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = m->matrix_data[i];
    } else {
      result->matrix_data[i] = 0;
    }
  }
  return result;
}

Matrix* relu_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying ReLU_prime activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
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

Matrix* tanh_activation(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Tanh activation to a %dx%d matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    result->matrix_data[i] = tanh(m->matrix_data[i]);
  }
  return result;
}

Matrix* tanh_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Tanh_prime activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    double tanh_val = tanh(m->matrix_data[i]);
    result->matrix_data[i] = 1.0 - pow(tanh_val, 2);
  }
  return result;
}

//============================
// Leaky ReLU Activation
//============================

// Without explicit definition, implemented in the functions below this
// This will assume the alpha
Matrix* leaky_relu(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Leaky ReLU activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = m->matrix_data[i];
    } else {
      result->matrix_data[i] = 0.01 * m->matrix_data[i];
    }
  }
  return result;
}

Matrix* leaky_relu_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Leaky ReLU_prime activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = 1;
    } else {
      result->matrix_data[i] = 0.01;
    }
  }
  return result;
}

// For when users may require more explicit defintions of alpha
Matrix* leaky_relu_with_alpha(Matrix* m, double leak_parameter) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  // If I converted a non acceptable value of alpha into 0.01, it would bring in
  // debug troubles.
  ASSERT(leak_parameter >= 0.0, "Alpha value must be non-negative.");

  LOG_INFO(
      "Applying Leaky ReLU with leak_parameter=%.2f activation function to a "
      "%dx%d "
      "matrix.",
      leak_parameter, m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > 0) {
      result->matrix_data[i] = m->matrix_data[i];
    } else {
      result->matrix_data[i] = leak_parameter * m->matrix_data[i];
    }
  }

  return result;
}

Matrix* leaky_relu_prime_with_alpha(Matrix* m, double leak_parameter) {
  ASSERT(m != NULL, "Input matrix for leaky_relu_prime is NULL.");
  ASSERT(leak_parameter >= 0.0, "Alpha value must be non-negative.");
  LOG_INFO("Applying Leaky ReLU with alpha=%.2f derivative to a %dx%d matrix.",
           leak_parameter, m->rows, m->cols);
  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;

  for (int i = 0; i < total_elements; i++) {
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

Matrix* sign_activation(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Sign activation to a %dx%d matrix.", m->rows, m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
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

Matrix* sign_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Sign_prime activation to a %dx%d matrix.", m->rows,
           m->cols);
  // The derivative of the sign function is 0 everywhere except at 0, where it
  // is undefined. For backpropagation, the derivative is commonly approximated
  // as 0.
  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    result->matrix_data[i] = 0.0;
  }
  return result;
}

//============================
// Identity Activation
//============================

Matrix* identity_activation(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Identity activation to a %dx%d matrix.", m->rows, m->cols);
  return copy_matrix(m);
}

Matrix* identity_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Identity_prime activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    result->matrix_data[i] = 1.0;
  }
  return result;
}

//============================
// Hard Tanh Activation
//============================

Matrix* hard_tanh(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Hard Tanh activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
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

Matrix* hard_tanh_prime(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  LOG_INFO("Applying Hard Tanh_prime activation to a %dx%d matrix.", m->rows,
           m->cols);

  Matrix* result = create_matrix(m->rows, m->cols);
  int total_elements = m->rows * m->cols;
  for (int i = 0; i < total_elements; i++) {
    if (m->matrix_data[i] > -1.0 && m->matrix_data[i] < 1.0) {
      result->matrix_data[i] = 1.0;
    } else {
      result->matrix_data[i] = 0.0;
    }
  }
  return result;
}
