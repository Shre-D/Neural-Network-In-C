/**
 * @file loss.c
 * @brief Implementations of loss functions and their gradients.
 */
#include "loss.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"
#include "utils.h"

// A small value to prevent log(0) errors.
#define EPSILON 1e-15

LossFunction get_loss_function(LossFunctionType type) {
  switch (type) {
    case MSE:
      return mean_squared_error;
    case CCE:
      return categorical_cross_entropy;
    case MAE:
      return mean_absolute_error;
    case BCE:
      return binary_cross_entropy;
    default:
      LOG_ERROR("Unknown loss function type.");
      return NULL;
  }
}

/**
 * @brief Returns a function pointer to the gradient of a specified loss
 * function.
 * @param type The type of the loss function gradient to retrieve.
 * @return A function pointer to the corresponding loss gradient function, or
 * NULL if the type is unknown.
 */
LossFunctionGrad get_loss_gradient(LossFunctionGradType type) {
  switch (type) {
    case MSE_GRAD:
      return mean_squared_error_gradient;
    case CCE_GRAD:
      return categorical_cross_entropy_gradient;
    case MAE_GRAD:
      return mean_absolute_error_gradient;
    case BCE_GRAD:
      return binary_cross_entropy_gradient;
    default:
      LOG_ERROR("Unknown loss gradient type.");
      return NULL;
  }
}

/**
 * @brief Computes the Mean Squared Error (MSE) loss between predicted and true
 * values.
 * @param y_hat A pointer to the Matrix of predicted values.
 * @param y A pointer to the Matrix of true values.
 * @return The calculated MSE loss.
 */
double mean_squared_error(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MSE: Matrices must have matching dimensions.");

  double loss = 0.0;
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    double diff = y_hat->matrix_data[i] - y->matrix_data[i];
    loss += pow(diff, 2);
  }

  return loss / total_elements;
}

/**
 * @brief Computes the Categorical Cross-Entropy (CCE) loss between predicted
 * and true values.
 * @param y_hat A pointer to the Matrix of predicted probabilities.
 * @param y A pointer to the Matrix of true one-hot encoded labels.
 * @return The calculated CCE loss.
 */
double categorical_cross_entropy(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Categorical Cross-Entropy: Matrices must have matching dimensions.");

  double loss = 0.0;
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    loss -= y->matrix_data[i] * log(y_hat->matrix_data[i] + EPSILON);
  }

  return loss / y_hat->rows;
}

/**
 * @brief Computes the Mean Absolute Error (MAE) loss between predicted and true
 * values.
 * @param y_hat A pointer to the Matrix of predicted values.
 * @param y A pointer to the Matrix of true values.
 * @return The calculated MAE loss.
 */
double mean_absolute_error(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MAE: Matrices must have matching dimensions.");

  double loss = 0.0;
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    loss += fabs(y_hat->matrix_data[i] - y->matrix_data[i]);
  }

  return loss / total_elements;
}

/**
 * @brief Computes the Binary Cross-Entropy (BCE) loss between predicted and
 * true values.
 * @param y_hat A pointer to the Matrix of predicted probabilities.
 * @param y A pointer to the Matrix of true binary labels.
 * @return The calculated BCE loss.
 */
double binary_cross_entropy(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Binary Cross-Entropy: Matrices must have matching dimensions.");

  double loss = 0.0;
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    loss -= y->matrix_data[i] * log(y_hat->matrix_data[i] + EPSILON) +
            (1 - y->matrix_data[i]) * log(1 - y_hat->matrix_data[i] + EPSILON);
  }

  return loss / total_elements;
}

/**
 * @brief Computes the gradient of the Mean Squared Error (MSE) loss with
 * respect to the predicted values.
 * @param y_hat A pointer to the Matrix of predicted values.
 * @param y A pointer to the Matrix of true values.
 * @return A new matrix containing the gradient of the MSE loss.
 */
Matrix* mean_squared_error_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MSE Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    gradient->matrix_data[i] =
        2.0 * (y_hat->matrix_data[i] - y->matrix_data[i]);
  }

  return gradient;
}

/**
 * @brief Computes the gradient of the Categorical Cross-Entropy (CCE) loss with
 * respect to the predicted probabilities.
 * @param y_hat A pointer to the Matrix of predicted probabilities.
 * @param y A pointer to the Matrix of true one-hot encoded labels.
 * @return A new matrix containing the gradient of the CCE loss.
 */
Matrix* categorical_cross_entropy_gradient(const Matrix* y_hat,
                                           const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "Categorical Cross-Entropy Gradient: Matrices must have matching "
         "dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    gradient->matrix_data[i] =
        -y->matrix_data[i] / (y_hat->matrix_data[i] + EPSILON);
  }

  return gradient;
}

/**
 * @brief Computes the gradient of the Mean Absolute Error (MAE) loss with
 * respect to the predicted values.
 * @param y_hat A pointer to the Matrix of predicted values.
 * @param y A pointer to the Matrix of true values.
 * @return A new matrix containing the gradient of the MAE loss.
 */
Matrix* mean_absolute_error_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(y_hat->rows == y->rows && y_hat->cols == y->cols,
         "MAE Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
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

/**
 * @brief Computes the gradient of the Binary Cross-Entropy (BCE) loss with
 * respect to the predicted probabilities.
 * @param y_hat A pointer to the Matrix of predicted probabilities.
 * @param y A pointer to the Matrix of true binary labels.
 * @return A new matrix containing the gradient of the BCE loss.
 */
Matrix* binary_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y) {
  ASSERT(
      y_hat->rows == y->rows && y_hat->cols == y->cols,
      "Binary Cross-Entropy Gradient: Matrices must have matching dimensions.");

  Matrix* gradient = create_matrix(y_hat->rows, y_hat->cols);
  size_t total_elements = y_hat->rows * y_hat->cols;

  for (size_t i = 0; i < total_elements; i++) {
    gradient->matrix_data[i] =
        (y_hat->matrix_data[i] - y->matrix_data[i]) /
        (y_hat->matrix_data[i] * (1 - y_hat->matrix_data[i]) + EPSILON);
  }

  return gradient;
}
