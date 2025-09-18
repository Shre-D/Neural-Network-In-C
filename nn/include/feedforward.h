#pragma once

#include "neural_network.h"

/**
 * @file feedforward.h
 * @brief Network creation and forward pass APIs.
 */

//==============================
// Forward Pass Functions
//==============================

/**
 * @brief Allocate and initialize a network with `num_layers` slots.
 * @param num_layers Number of layers to allocate.
 * @return A new, empty network with allocated layer slots.
 */
NeuralNetwork* create_network(size_t num_layers);

/**
 * @brief Free a network and its associated resources.
 * @param nn The network to free.
 */
void free_network(NeuralNetwork* nn);

/**
 * @brief Run the forward pass and cache intermediates for backprop.
 * @param nn Network pointer (non-NULL).
 * @param input Input matrix (batch_size x input_features).
 * @return Output activation of the last layer (batch_size x output_features).
 *         Caller owns and must free.
 */
Matrix* feedforward(NeuralNetwork* nn, const Matrix* input);
