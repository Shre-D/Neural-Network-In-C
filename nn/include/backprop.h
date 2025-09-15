#pragma once

#include "loss.h"
#include "neural_network.h"

/**
 * @file backprop.h
 * @brief Backpropagation and gradient computation APIs.
 */

//============================
// Functions for Backpropagation
//============================

/**
 * @brief Compute layer-wise deltas and cache them for gradient evaluation.
 * @param nn Pointer to network.
 * @param y_true Ground-truth labels/targets.
 * @param loss_func Loss function (optional here; for monitoring).
 * @param loss_func_grad Gradient of loss w.r.t predictions (required).
 */
void backpropagate(NeuralNetwork* nn, const Matrix* y_true,
                   LossFunction loss_func, LossFunctionGrad loss_func_grad);

/** @brief Calculate weight gradient for a specific layer. */
Matrix* calculate_weight_gradient(const Cache* cache, size_t layer_index,
                                  size_t total_layers);

/** @brief Calculate bias gradient for a specific layer. */
Matrix* calculate_bias_gradient(const Cache* cache, size_t layer_index,
                                size_t total_layers);
