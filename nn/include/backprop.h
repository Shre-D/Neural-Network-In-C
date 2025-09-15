#pragma once

#include "loss.h"
#include "neural_network.h"

//============================
// Functions for Backpropagation
//============================

void backpropagate(NeuralNetwork* nn, const Matrix* y_true,
                   LossFunction loss_func, LossFunctionGrad loss_func_grad);

// Calculates the gradient of a single layer's weights.
Matrix* calculate_weight_gradient(const Cache* cache, int layer_index,
                                  int total_layers);

// Calculates the gradient of a single layer's bias.
Matrix* calculate_bias_gradient(const Cache* cache, int layer_index,
                                int total_layers);
