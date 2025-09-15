#pragma once

#include "activation.h"
#include "cache.h"
#include "linalg.h"

/**
 * @file neural_network.h
 * @brief Neural network layer and network structures.
 */

//==============================
// Neural Network Layer Struct
//==============================

/** @brief Function pointer type for matrix-wise activations. */
typedef Matrix* (*ActivationFunc)(Matrix*);

/**
 * @brief Fully connected layer parameters and activation.
 */
typedef struct {
  Matrix* weights; /**< Weight matrix (D_in×D_out). */
  Matrix* bias;    /**< Bias vector as (1×D_out). */

  ActivationFunc activation; /**< Activation function for this layer. */

  /** The leak parameter for Leaky ReLU activation. */
  double leak_parameter;
} Layer;

//==============================
// Neural Network Struct
//==============================

/**
 * @brief Neural network composed of sequential fully connected layers.
 */
typedef struct {
  Layer** layers; /**< Array of layer pointers (length = num_layers). */
  int num_layers; /**< Number of layers. */

  /** Caches intermediate forward/backward values. */
  Cache* cache;
} NeuralNetwork;
