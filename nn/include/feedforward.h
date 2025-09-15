#pragma once

#include "neural_network.h"

/**
 * @file feedforward.h
 * @brief Network creation and forward pass APIs.
 */

//==============================
// Forward Pass Functions
//==============================

/** @brief Allocate and initialize a network with `num_layers` slots. */
NeuralNetwork* create_network(int num_layers);

/** @brief Free a network and its associated resources. */
void free_network(NeuralNetwork* nn);

/**
 * @brief Run the forward pass and cache intermediates for backprop.
 * @param nn Network pointer (non-NULL).
 * @param input Input matrix matching first layer's expected shape.
 * @return Output activation of the last layer. Caller owns and must free.
 */
Matrix* feedforward(NeuralNetwork* nn, const Matrix* input);
