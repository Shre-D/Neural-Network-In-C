#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"
#include "loss.h"
#include "utils.h"

// A small value to prevent log(0) errors.
#define EPSILON 1e-15

double mean_squared_error(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MSE: Matrices must have matching dimensions.");

  double loss = 0.0;
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    double diff = y_hat->matrix_data[i] - y->matrix_data[i];
    loss += pow(diff, 2);
  }

  return loss / total_elements;
}

double categorical_cross_entropy(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Categorical Cross-Entropy: Matrices must have matching dimensions.");

  double loss = 0.0;
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    loss -= y->matrix_data[i] * log(y_hat->matrix_data[i] + EPSILON);
  }

  return loss / y_hat->rows;
}

double mean_absolute_error(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MAE: Matrices must have matching dimensions.");

  double loss = 0.0;
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    loss += fabs(y_hat->matrix_data[i] - y->matrix_data[i]);
  }

  return loss / total_elements;
}

double binary_cross_entropy(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Binary Cross-Entropy: Matrices must have matching dimensions.");

  double loss = 0.0;
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    loss -= y->matrix_data[i] * log(y_hat->matrix_data[i] + EPSILON) +
            (1 - y->matrix_data[i]) * log(1 - y_hat->matrix_data[i] + EPSILON);
  }

  return loss / total_elements;
}

Matrix* mean_squared_error_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MSE Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    gradient->matrix_data[i] = 2.0 * (y_hat->matrix_data[i] - y->matrix_data[i]);
  }

  return gradient;
}

Matrix* categorical_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Categorical Cross-Entropy Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    gradient->matrix_data[i] = -y->matrix_data[i] / (y_hat->matrix_data[i] + EPSILON);
  }

  return gradient;
}

Matrix* mean_absolute_error_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MAE Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    if (y_hat->matrix_data[i] > y->matrix_data[i]) {
      gradient->matrix_data[i] = 1.0;
    } else if (y_hat->matrix_data[i] < y->matrix_data[i]) {
      gradient->matrix_data[i] = -1.0;
    } else {
      gradient->matrix_data[i] = 0.0;
    }
  }

  return gradient;
}

Matrix* binary_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Binary Cross-Entropy Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  int total_elements = y_hat->rows * y_hat->cols;

  for (int i = 0; i < total_elements; i++) {
    gradient->matrix_data[i] = (y_hat->matrix_data[i] - y->matrix_data[i]) /
                               (y_hat->matrix_data[i] * (1 - y_hat->matrix_data[i]) + EPSILON);
  }

  return gradient;
}
