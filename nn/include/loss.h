#pragma once

#include "linalg.h"
#include "utils.h"

typedef enum {
  MSE,  // Mean Squared Error
  CCE,  // Categorical Cross-Entropy
  MAE,  // Mean Absolute Error
  BCE,  // Binary Cross-Entropy
} LossFunctionType;

typedef enum {
  MSE_GRAD,  // Mean Squared Error Gradient
  CCE_GRAD,  // Categorical Cross-Entropy Gradient
  MAE_GRAD,  // Mean Absolute Error Gradient
  BCE_GRAD,  // Binary Cross-Entropy Gradient
} LossFunctionGradType;

typedef double (*LossFunction)(const Matrix* y_hat, const Matrix* y);

typedef Matrix* (*LossFunctionGrad)(const Matrix* y_hat, const Matrix* y);

//=====================
// Loss Functions
//=====================

double mean_squared_error(const Matrix* y_hat, const Matrix* y);
double categorical_cross_entropy(const Matrix* y_hat, const Matrix* y);
double mean_absolute_error(const Matrix* y_hat, const Matrix* y);
double binary_cross_entropy(const Matrix* y_hat, const Matrix* y);

//==============================
// Loss Function Gradients
//==============================

Matrix* mean_squared_error_gradient(const Matrix* y_hat, const Matrix* y);
Matrix* categorical_cross_entropy_gradient(const Matrix* y_hat,
                                           const Matrix* y);
Matrix* mean_absolute_error_gradient(const Matrix* y_hat, const Matrix* y);
Matrix* binary_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y);
