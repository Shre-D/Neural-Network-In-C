/**
 * @file feedforward.c
 * @brief Network allocation and forward pass with caching of intermediates.
 */
#include "feedforward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "linalg.h"
#include "neural_network.h"
#include "utils.h"

NeuralNetwork* create_network(size_t num_layers) {
  NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
  if (nn == NULL) {
    LOG_ERROR("Memory allocation failed for Neural Network struct.");
    return NULL;
  }

  nn->layers = (Layer**)malloc(sizeof(Layer*) * num_layers);
  if (nn->layers == NULL) {
    LOG_ERROR("Memory allocation failed for layers array.");
    free(nn);
    return NULL;
  }
  // Initialize layer pointers to NULL
  for (size_t i = 0; i < num_layers; i++) {
    nn->layers[i] = NULL;
  }

  nn->num_layers = num_layers;
  nn->cache = create_cache();
  if (nn->cache == NULL) {
    LOG_ERROR("Failed to initialize cache.");
    free(nn->layers);
    free(nn);
    return NULL;
  }
  return nn;
}

/**
 * @brief Frees all memory associated with a neural network.
 * This includes layers, weights, biases, and the cache.
 * @param nn A pointer to the NeuralNetwork structure to be freed.
 */
void free_network(NeuralNetwork* nn) {
  if (nn == NULL) {
    return;
  }

  if (nn->layers != NULL) {
    for (size_t i = 0; i < nn->num_layers; i++) {
      if (nn->layers[i] != NULL) {
        if (nn->layers[i]->weights != NULL) {
          free_matrix(nn->layers[i]->weights);
        }
        if (nn->layers[i]->bias != NULL) {
          free_matrix(nn->layers[i]->bias);
        }
        free(nn->layers[i]);
      }
    }
    free(nn->layers);
  }
  if (nn->cache != NULL) {
    free_cache(nn->cache);
  }
  free(nn);
}

/**
 * @brief Performs a forward pass through the neural network.
 * Computes the output of the network for a given input and caches intermediate
 * values.
 * @param nn A pointer to the NeuralNetwork structure.
 * @param input A pointer to the input Matrix.
 * @return A new matrix containing the output of the last layer of the network.
 * The caller is responsible for freeing this matrix.
 */
Matrix* feedforward(const NeuralNetwork* nn, const Matrix* input) {
  ASSERT(nn != NULL, "Neural Network pointer cannot be NULL.");
  ASSERT(input != NULL, "Input matrix cannot be NULL.");
  ASSERT(input->cols == nn->layers[0]->weights->rows,
         "Input dimensions must match network dimensions.");

  Matrix* current_output = copy_matrix(input);
  ASSERT(current_output != NULL, "Failed to copy input matrix.");

  cache_put(nn->cache, "input", copy_matrix(current_output));

  for (size_t i = 0; i < nn->num_layers; i++) {
    Layer* current_layer = nn->layers[i];
    ASSERT(current_output->cols == current_layer->weights->rows,
           "Shape mismatch: output cols != weights rows.");

    Matrix* z_linear = dot_matrix(current_output, current_layer->weights);
    ASSERT(z_linear != NULL, "Dot product failed.");
    ASSERT(z_linear->rows == current_output->rows &&
               z_linear->cols == current_layer->weights->cols,
           "Unexpected shape from dot product.");

    Matrix* z = add_bias_to_matrix(z_linear, current_layer->bias);
    ASSERT(z != NULL, "Bias add failed.");
    ASSERT(z->rows == z_linear->rows && z->cols == z_linear->cols,
           "Unexpected shape from bias add.");

    // Cache the intermediate pre-activation value (z).
    char z_key[32];
    sprintf(z_key, "z_%zu", i);
    cache_put(nn->cache, z_key, copy_matrix(z));

    Matrix* a = NULL;
    switch (current_layer->activation_type) {
      case SIGMOID:
        a = sigmoid(z);
        break;
      case RELU:
        a = relu(z);
        break;
      case TANH:
        a = tanh_activation(z);
        break;
      case LEAKY_RELU:
        a = leaky_relu(z, current_layer->leak_parameter);
        break;
      case SIGN:
        a = sign_activation(z);
        break;
      case IDENTITY:
        a = identity_activation(z);
        break;
      case HARD_TANH:
        a = hard_tanh(z);
        break;
      case SOFTMAX:
        a = softmax(z);
        break;
      default:
        LOG_WARN("Unknown activation function, defaulting to identity.");
        a = identity_activation(z);
        break;
    }
    ASSERT(a != NULL, "Activation failed.");
    ASSERT(a->rows == z->rows && a->cols == z->cols,
           "Unexpected shape from activation.");

    char a_key[32];
    sprintf(a_key, "a_%zu", i);
    cache_put(nn->cache, a_key, copy_matrix(a));

    free_matrix(z_linear);
    free_matrix(z);
    free_matrix(current_output);
    current_output = a;
  }

  return current_output;
}
