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
typedef struct _Layer {
  Matrix* weights; /**< Weight matrix (D_in×D_out). */
  Matrix* bias;    /**< Bias vector as (1×D_out). */

  activation_function
      activation_type; /**< Type of activation function for this layer. */

  /** The leak parameter for Leaky ReLU activation. */
  double leak_parameter;
} Layer;

//==============================
// Neural Network Struct
//==============================

/**
 * @brief Neural network composed of sequential fully connected layers.
 */
typedef struct _NeuralNetwork {
  Layer** layers;    /**< Array of layer pointers (length = num_layers). */
  size_t num_layers; /**< Number of layers. */

  /** Caches intermediate forward/backward values. */
  Cache* cache;
} NeuralNetwork;

/**
 * @brief Prints a summary of the neural network's architecture.
 * @param stream The output stream.
 * @param nn The neural network to summarize.
 */
void fprint_network_summary(FILE* stream, const NeuralNetwork* nn);

/**
 * @brief Logs the training progress to a file.
 * @param stream The output stream.
 * @param epoch The current epoch.
 * @param epochs The total number of epochs.
 * @param loss The training loss.
 */
void flog_training_progress(FILE* stream, int epoch, int epochs, double loss);
