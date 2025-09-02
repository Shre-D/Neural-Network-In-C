#include "feedforward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"
#include "neural_network.h"
#include "utils.h"

// Macro to handle memory allocation checks.
#define CHECK_ALLOC(ptr)                    \
  if (ptr == NULL) {                        \
    LOG_ERROR("Memory allocation failed."); \
    return NULL;                            \
  }

NeuralNetwork* create_network(int num_layers) {
  NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
  CHECK_ALLOC(nn);

  nn->layers = (Layer**)malloc(sizeof(Layer*) * num_layers);
  CHECK_ALLOC(nn->layers);

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
    for (int i = 0; i < nn->num_layers; i++) {
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

  // Cache the input for the backpropagation algorithm.
  put_matrix(nn->cache, "input", current_output);

  for (int i = 0; i < nn->num_layers; i++) {
    Matrix* z = dot_matrix(current_output, nn->layers[i]->weights);

    // Add bias
    add_matrix(z, nn->layers[i]->bias);

    // Cache the intermediate value (z).
    char z_key[32];
    sprintf(z_key, "z_%d", i);
    put_matrix(nn->cache, z_key, z);

    Matrix* a = apply_onto_matrix(nn->layers[i]->activation, z);

    // Cache the activated output (a).
    char a_key[32];
    sprintf(a_key, "a_%d", i);
    put_matrix(nn->cache, a_key, a);

    free_matrix(z);  // We no longer need z, as we have cached it.
    free_matrix(current_output);
    current_output = a;
  }

  return current_output;
}
