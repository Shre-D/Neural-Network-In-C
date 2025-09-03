#pragma once

#include "neural_network.h"

//==============================
// Forward Pass Functions
//==============================

NeuralNetwork* create_network(int num_layers);

void free_network(NeuralNetwork* nn);

// It caches the intermediate results for the backpropagation algorithm.
Matrix* feedforward(NeuralNetwork* nn, const Matrix* input);
