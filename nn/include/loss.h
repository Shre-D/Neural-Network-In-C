#pragma once

#include "linalg.h"
#include "utils.h"

/**
 * @file loss.h
 * @brief Loss functions and their gradients.
 *
 * Provides scalar loss computations and matrix-valued gradients with respect to
 * predictions `y_hat`. All gradient functions return newly allocated matrices;
 * callers own and must free them.
 */

/** @brief Enumerates supported loss functions. */
typedef enum {
  MSE,  /**< Mean Squared Error */
  CCE,  /**< Categorical Cross-Entropy */
  MAE,  /**< Mean Absolute Error */
  BCE,  /**< Binary Cross-Entropy */
} LossFunctionType;

/** @brief Enumerates supported loss gradients. */
typedef enum {
  MSE_GRAD,  /**< Mean Squared Error Gradient */
  CCE_GRAD,  /**< Categorical Cross-Entropy Gradient */
  MAE_GRAD,  /**< Mean Absolute Error Gradient */
  BCE_GRAD,  /**< Binary Cross-Entropy Gradient */
} LossFunctionGradType;

/** @brief Function pointer type for scalar loss. */
typedef double (*LossFunction)(const Matrix* y_hat, const Matrix* y);

/** @brief Function pointer type for loss gradient. */
typedef Matrix* (*LossFunctionGrad)(const Matrix* y_hat, const Matrix* y);

//=====================
// Loss Functions
//=====================

/** @brief Mean Squared Error (MSE). */
double mean_squared_error(const Matrix* y_hat, const Matrix* y);
/** @brief Categorical Cross-Entropy (CCE). */
double categorical_cross_entropy(const Matrix* y_hat, const Matrix* y);
/** @brief Mean Absolute Error (MAE). */
double mean_absolute_error(const Matrix* y_hat, const Matrix* y);
/** @brief Binary Cross-Entropy (BCE). */
double binary_cross_entropy(const Matrix* y_hat, const Matrix* y);

//==============================
// Loss Function Gradients
//==============================

/** @brief Gradient of MSE with respect to y_hat. */
Matrix* mean_squared_error_gradient(const Matrix* y_hat, const Matrix* y);
/** @brief Gradient of CCE with respect to y_hat. */
Matrix* categorical_cross_entropy_gradient(const Matrix* y_hat,
                                           const Matrix* y);
/** @brief Gradient of MAE with respect to y_hat. */
Matrix* mean_absolute_error_gradient(const Matrix* y_hat, const Matrix* y);
/** @brief Gradient of BCE with respect to y_hat. */
Matrix* binary_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y);
