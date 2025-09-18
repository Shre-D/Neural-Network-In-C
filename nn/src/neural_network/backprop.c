/**
 * @file backprop.c
 * @brief Backpropagation implementation and gradient helpers.
 */
#include "backprop.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "activation.h"
#include "cache.h"
#include "linalg.h"
#include "neural_network.h"
#include "utils.h"

// Select and compute activation derivative for a layer given its pre-activation
// input z. Handles common activations and leaky ReLU with optional alpha.
/**
 * @brief Selects and computes the derivative of the activation function for a
 * given layer.
 * @param layer A pointer to the Layer structure containing the activation type
 * and parameters.
 * @param z A pointer to the pre-activation matrix (input to the activation
 * function).
 * @return A new matrix containing the element-wise derivative of the activation
 * function applied to z.
 */
static Matrix* activation_derivative_for_layer(const Layer* layer, Matrix* z) {
  ASSERT(layer != NULL, "Layer cannot be NULL.");
  ASSERT(z != NULL, "Pre-activation matrix z cannot be NULL.");

  switch (layer->activation_type) {
    case SIGMOID:
      return sigmoid_prime(z);
    case RELU:
      return relu_prime(z);
    case TANH:
      return tanh_prime(z);
    case LEAKY_RELU:
      return leaky_relu_prime(z, layer->leak_parameter);
    case SIGN:
      return sign_prime(z);
    case IDENTITY:
      return identity_prime(z);
    case HARD_TANH:
      return hard_tanh_prime(z);
    default:
      LOG_WARN(
          "Unknown activation function, defaulting derivative to identity.");
      return identity_prime(z);
  }
}

void backpropagate(NeuralNetwork* nn, const Matrix* y_true,
                   LossFunctionType loss_type,
                   LossFunctionGrad loss_func_grad) {
  ASSERT(nn != NULL, "Neural Network pointer cannot be NULL.");
  ASSERT(nn->cache != NULL, "Cache cannot be NULL.");
  ASSERT(y_true != NULL, "Ground truth matrix cannot be NULL.");
  ASSERT(loss_func_grad != NULL, "Loss gradient function cannot be NULL.");

  size_t last_index = nn->num_layers - 1;
  Layer* last_layer = nn->layers[last_index];

  // Get y_hat from cache (activation of last layer)
  char a_last_key[32];
  sprintf(a_last_key, "a_%zu", last_index);
  Matrix* y_hat = cache_get(nn->cache, a_last_key);
  ASSERT(y_hat != NULL, "Cached prediction (y_hat) not found.");

  Matrix* delta_last;
  // Special case for Softmax with CCE
  if (last_layer->activation_type == SOFTMAX && loss_type == CCE) {
    delta_last = subtract_matrix(y_hat, (Matrix*)y_true);
  } else {
    // dL/da for output layer
    Matrix* dL_da = loss_func_grad(y_hat, y_true);
    ASSERT(dL_da != NULL, "Loss gradient returned NULL.");

    // delta for output layer: dL/dz = dL/da .* a'(z)
    char z_last_key[32];
    sprintf(z_last_key, "z_%zu", last_index);
    Matrix* z_last = cache_get(nn->cache, z_last_key);
    ASSERT(z_last != NULL, "Cached z for last layer not found.");

    Matrix* act_prime_last =
        activation_derivative_for_layer(last_layer, z_last);
    delta_last = multiply_matrix(dL_da, act_prime_last);
    ASSERT(delta_last != NULL, "Failed to compute delta for last layer.");

    free_matrix(dL_da);
    free_matrix(z_last);
    free_matrix(act_prime_last);
  }

  char delta_last_key[32];
  sprintf(delta_last_key, "delta_%zu", last_index);
  cache_put(nn->cache, delta_last_key, delta_last);

  // Clean up temporaries for last layer
  free_matrix(y_hat);

  // Backpropagate through hidden layers
  for (size_t i = last_index - 1; i != SIZE_MAX; i--) {
    // delta_{i} = (delta_{i+1} dot W_{i+1}^T) .* a'_i(z_i)
    char delta_next_key[32];
    sprintf(delta_next_key, "delta_%zu", i + 1);
    Matrix* delta_next = cache_get(nn->cache, delta_next_key);
    ASSERT(delta_next != NULL, "Cached delta for next layer not found.");

    Matrix* W_next = nn->layers[i + 1]->weights;
    ASSERT(W_next != NULL, "Weights for next layer cannot be NULL.");

    Matrix* W_next_T = transpose_matrix(W_next);
    Matrix* propagated = dot_matrix(delta_next, W_next_T);

    char z_key[32];
    sprintf(z_key, "z_%zu", i);
    Matrix* z_i = cache_get(nn->cache, z_key);
    ASSERT(z_i != NULL, "Cached z for layer not found.");
    Matrix* act_prime_i = activation_derivative_for_layer(nn->layers[i], z_i);

    Matrix* delta_i = multiply_matrix(propagated, act_prime_i);
    ASSERT(delta_i != NULL, "Failed to compute delta for layer.");

    char delta_i_key[32];
    sprintf(delta_i_key, "delta_%zu", i);
    cache_put(nn->cache, delta_i_key, delta_i);

    // Clean up
    free_matrix(delta_next);
    free_matrix(W_next_T);
    free_matrix(propagated);
    free_matrix(z_i);
    free_matrix(act_prime_i);
  }
}

/**
 * @brief Calculates the gradient of the weights for a specific layer during
 * backpropagation.
 * @param cache A pointer to the Cache containing intermediate values.
 * @param layer_index The index of the current layer.
 * @param total_layers The total number of layers in the neural network.
 * @return A new matrix representing the gradient of the weights for the
 * specified layer.
 */
Matrix* calculate_weight_gradient(const Cache* cache, size_t layer_index,
                                  size_t total_layers) {
  ASSERT(cache != NULL, "Cache cannot be NULL.");
  ASSERT(layer_index < total_layers, "layer_index out of bounds.");

  // Get activation of previous layer (or input)
  Matrix* a_prev = NULL;
  if (layer_index == 0) {
    a_prev = cache_get((Cache*)cache, "input");
  } else {
    char a_prev_key[32];
    sprintf(a_prev_key, "a_%zu", layer_index - 1);
    a_prev = cache_get((Cache*)cache, a_prev_key);
  }
  ASSERT(a_prev != NULL, "Cached previous activation/input not found.");

  // Get delta for current layer
  char delta_key[32];
  sprintf(delta_key, "delta_%zu", layer_index);
  Matrix* delta_i = cache_get((Cache*)cache, delta_key);
  ASSERT(delta_i != NULL, "Cached delta for layer not found.");

  Matrix* a_prev_T = transpose_matrix(a_prev);
  Matrix* grad_W = dot_matrix(a_prev_T, delta_i);

  // Clean up
  free_matrix(a_prev);
  free_matrix(delta_i);
  free_matrix(a_prev_T);

  return grad_W;
}

/**
 * @brief Calculates the gradient of the biases for a specific layer during
 * backpropagation.
 * @param cache A pointer to the Cache containing intermediate values.
 * @param layer_index The index of the current layer.
 * @param total_layers The total number of layers in the neural network.
 * @return A new matrix representing the gradient of the biases for the
 * specified layer.
 */
Matrix* calculate_bias_gradient(const Cache* cache, size_t layer_index,
                                size_t total_layers) {
  ASSERT(cache != NULL, "Cache cannot be NULL.");
  ASSERT(layer_index < total_layers, "layer_index out of bounds.");

  char delta_key[32];
  sprintf(delta_key, "delta_%zu", layer_index);
  Matrix* delta_i = cache_get((Cache*)cache, delta_key);
  ASSERT(delta_i != NULL, "Cached delta for layer not found.");

  Matrix* db = sum_matrix_columns(delta_i);
  free_matrix(delta_i);

  return db;
}
