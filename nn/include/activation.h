#pragma once

#include "linalg.h"

//============================
// Activation Functions
//============================

// We'll need the derivatives of these functions in the backprop algo
// So we'll define functions for derivatives here too.

Matrix* sigmoid(Matrix* m);
Matrix* sigmoid_prime(Matrix* m);

Matrix* relu(Matrix* m);
Matrix* relu_prime(Matrix* m);

Matrix* tanh_activation(Matrix* m);
Matrix* tanh_prime(Matrix* m);

Matrix* leaky_relu(Matrix* m);
Matrix* leaky_relu_prime(Matrix* m);

// Allows the user to specify a custom alpha value
Matrix* leaky_relu_with_alpha(Matrix* m, double leak_parameter);
Matrix* leaky_relu_prime_with_alpha(Matrix* m, double leak_parameter);

Matrix* sign_activation(Matrix* m);
Matrix* sign_prime(Matrix* m);

Matrix* identity_activation(Matrix* m);
Matrix* identity_prime(Matrix* m);

Matrix* hard_tanh(Matrix* m);
Matrix* hard_tanh_prime(Matrix* m);
