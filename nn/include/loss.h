#pragma once

#include "linalg.h"
#include "utils.h"

//==============================
// My first loss function library
// I'm going to define common loss functions and their gradients here
//==============================

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
Matrix* categorical_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y);
Matrix* mean_absolute_error_gradient(const Matrix* y_hat, const Matrix* y);
Matrix* binary_cross_entropy_gradient(const Matrix* y_hat, const Matrix* y);
