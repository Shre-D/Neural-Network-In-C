/**
 * @file activation.h
 * @brief Header for activation functions used in neural networks.
 *
 * This file defines an enumeration for various activation functions and
 * declares their corresponding matrix-based functions and their derivatives.
 * These functions are crucial for introducing non-linearity into neural
 * networks.
 */

#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include <stddef.h>

#include "linalg.h"  // Assumes Matrix struct is defined here

/**
 * @brief Enum for different activation functions.
 *
 * This enumeration lists the types of activation functions supported
 * within the neural network framework. Each enumerator corresponds to
 * a specific non-linear function applied to the output of a layer.
 */
typedef enum {
  RELU,       /**< Rectified Linear Unit activation. */
  SIGMOID,    /**< Sigmoid activation. */
  SOFTMAX,    /**< Softmax activation, typically used in the output layer for
                 multi-class classification. */
  TANH,       /**< Hyperbolic Tangent activation. */
  LEAKY_RELU, /**< Leaky Rectified Linear Unit activation. */
  SIGN,       /**< Sign activation. */
  IDENTITY,   /**< Identity activation. */
  HARD_TANH   /**< Hard Tanh activation. */
} activation_function;

// Activation functions

/**
 * @brief Applies the Sigmoid activation function element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the sigmoid function applied to each element.
 */
Matrix* sigmoid(Matrix* m);

/**
 * @brief Applies the ReLU (Rectified Linear Unit) activation function
 * element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the ReLU function applied to each element.
 */
Matrix* relu(Matrix* m);

/**
 * @brief Applies the Hyperbolic Tangent (tanh) activation function
 * element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the tanh function applied to each element.
 */
Matrix* tanh_activation(Matrix* m);

/**
 * @brief Applies the Leaky ReLU activation function element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @param leak_parameter The leak parameter (alpha) for the Leaky ReLU.
 * @return A new Matrix with the Leaky ReLU function applied to each element.
 */
Matrix* leaky_relu(Matrix* m, double leak_parameter);

/**
 * @brief Applies the Sign activation function element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the Sign function applied to each element.
 */
Matrix* sign_activation(Matrix* m);

/**
 * @brief Applies the Identity activation function element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix that is a copy of the input matrix.
 */
Matrix* identity_activation(Matrix* m);

/**
 * @brief Applies the Hard Tanh activation function element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the Hard Tanh function applied to each element.
 */
Matrix* hard_tanh(Matrix* m);

/**
 * @brief Applies the Softmax activation function to a matrix.
 * This function is typically used in the output layer of a neural network for
 * multi-class classification. It normalizes the input values into a probability
 * distribution.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the Softmax function applied.
 */
Matrix* softmax(Matrix* m);

// Derivatives of activation functions

/**
 * @brief Computes the derivative of the Sigmoid activation function
 * element-wise to a matrix.
 * @param m A pointer to the input Matrix (output of the sigmoid function).
 * @return A new Matrix with the sigmoid derivative applied to each element.
 */
Matrix* sigmoid_prime(Matrix* m);

/**
 * @brief Computes the derivative of the ReLU activation function element-wise
 * to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the ReLU derivative applied to each element.
 */
Matrix* relu_prime(Matrix* m);

/**
 * @brief Computes the derivative of the Hyperbolic Tangent (tanh) activation
 * function element-wise to a matrix.
 * @param m A pointer to the input Matrix (output of the tanh function).
 * @return A new Matrix with the tanh derivative applied to each element.
 */
Matrix* tanh_prime(Matrix* m);

/**
 * @brief Computes the derivative of the Leaky ReLU activation function
 * element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @param leak_parameter The leak parameter (alpha) for the Leaky ReLU.
 * @return A new Matrix with the Leaky ReLU derivative applied to each element.
 */
Matrix* leaky_relu_prime(Matrix* m, double leak_parameter);

/**
 * @brief Computes the derivative of the Sign activation function element-wise
 * to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the Sign derivative applied to each element.
 */
Matrix* sign_prime(Matrix* m);

/**
 * @brief Computes the derivative of the Identity activation function
 * element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix of the same size as the input, with all elements set
 * to 1.0.
 */
Matrix* identity_prime(Matrix* m);

/**
 * @brief Computes the derivative of the Hard Tanh activation function
 * element-wise to a matrix.
 * @param m A pointer to the input Matrix.
 * @return A new Matrix with the Hard Tanh derivative applied to each element.
 */
Matrix* hard_tanh_prime(Matrix* m);

/**
 * @brief Computes the derivative of the Softmax activation function
 * element-wise to a matrix. This is typically used in conjunction with a loss
 * function like cross-entropy where the derivative simplifies to output * (1 -
 * output).
 * @param m A pointer to the input Matrix (output of the softmax function).
 * @return A new Matrix with the Softmax derivative applied to each element.
 */
Matrix* softmax_prime(Matrix* m);

#endif  // NN_ACTIVATION_H
/**
 * @brief Converts an activation function enum to its string representation.
 * @param func The activation function enum.
 * @return A string representing the activation function.
 */
const char* activation_to_string(activation_function func);