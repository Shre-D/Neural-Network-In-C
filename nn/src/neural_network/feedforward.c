/**
 * @file feedforward.c
 * @brief Network allocation and forward pass with caching of intermediates.
 */
#include "feedforward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  nn->num_layers = num_layers;
  nn->cache = init_cache();
  if (nn->cache == NULL) {
    LOG_ERROR("Failed to initialize cache.");
    free(nn->layers);
    free(nn);
    return NULL;
  }
  return nn;
}

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
    clear_cache(nn->cache);
    free(nn->cache);
  }
  free(nn);
}

Matrix* feedforward(NeuralNetwork* nn, const Matrix* input) {
  ASSERT(nn != NULL, "Neural Network pointer cannot be NULL.");
  ASSERT(input != NULL, "Input matrix cannot be NULL.");
  ASSERT(input->rows == nn->layers[0]->weights->cols,
         "Input dimensions must match network dimensions.");

  Matrix* current_output = copy_matrix(input);

  put_matrix(nn->cache, "input", current_output);

  for (size_t i = 0; i < nn->num_layers; i++) {
    Matrix* z_linear = dot_matrix(current_output, nn->layers[i]->weights);
    // Bias add returns a new matrix; capture it to avoid dropping the result.
    Matrix* z = add_matrix(z_linear, nn->layers[i]->bias);

    // Cache the intermediate pre-activation value (z).
    char z_key[32];
    sprintf(z_key, "z_%zu", i);
    put_matrix(nn->cache, z_key, z);

    Matrix* a = NULL;
    // Handle special-case leaky ReLU with custom alpha.
    if (nn->layers[i]->activation == leaky_relu_with_alpha) {
      a = leaky_relu_with_alpha(z, nn->layers[i]->leak_parameter);
    } else {
      a = nn->layers[i]->activation(z);
    }

    char a_key[32];
    sprintf(a_key, "a_%zu", i);
    put_matrix(nn->cache, a_key, a);

    free_matrix(z_linear);
    free_matrix(z);
    free_matrix(current_output);
    current_output = a;
  }

  return current_output;
}
