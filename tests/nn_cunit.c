/**
 * @file nn_cunit.c
 * @brief CUnit tests for the neural network core functionalities.
 */

#include <CUnit/Basic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "backprop.h"
#include "feedforward.h"
#include "linalg.h"
#include "loss.h"
#include "neural_network.h"
#include "test_utils.h"
#include "utils.h"

/**
 * @brief Tests the sigmoid activation function.
 * Verifies that the sigmoid function produces correct outputs for given inputs.
 */
void test_sigmoid() {
  Matrix* m = create_matrix(1, 2);
  Matrix* expected = create_matrix(1, 2);
  m->matrix_data[0] = 0;
  m->matrix_data[1] = 1;
  expected->matrix_data[0] = 0.5;
  expected->matrix_data[1] = 1.0 / (1.0 + exp(-1.0));  // approx 0.731

  Matrix* result = sigmoid(m);
  CU_ASSERT_TRUE(compare_matrices(result, expected, 1e-9));

  free_matrix(m);
  free_matrix(expected);
  free_matrix(result);
}

/**
 * @brief Tests the creation and freeing of a neural network.
 * Verifies that `create_network` allocates memory correctly and `free_network`
 * deallocates it without issues.
 */
void test_create_free_network() {
  NeuralNetwork* nn = create_network(2);
  CU_ASSERT_PTR_NOT_NULL(nn);
  CU_ASSERT_EQUAL(nn->num_layers, 2);
  CU_ASSERT_PTR_NOT_NULL(nn->cache);
  free_network(nn);
}

/**
 * @brief Tests a simple feedforward pass through a neural network.
 * Sets up a basic network and verifies the output of a forward pass.
 */
void test_feedforward_simple() {
  NeuralNetwork* nn = create_network(1);
  CU_ASSERT_PTR_NOT_NULL(nn);

  // Create a simple layer
  Layer* layer = (Layer*)malloc(sizeof(Layer));
  CU_ASSERT_PTR_NOT_NULL(layer);
  layer->weights = create_matrix(2, 1);  // 2 inputs, 1 output
  layer->weights->matrix_data[0] = 0.5;
  layer->weights->matrix_data[1] = 0.5;
  layer->bias = create_matrix(1, 1);
  layer->bias->matrix_data[0] = 0.1;
  layer->activation_type = SIGMOID;
  layer->leak_parameter = 0.0;  // Not used for sigmoid
  nn->layers[0] = layer;

  // Input matrix
  Matrix* input = create_matrix(1, 2);  // 1 sample, 2 features
  input->matrix_data[0] = 1.0;
  input->matrix_data[1] = 1.0;

  // Expected output: sigmoid((1*0.5) + (1*0.5) + 0.1) = sigmoid(1.1)
  Matrix* expected_output = create_matrix(1, 1);
  expected_output->matrix_data[0] = 1.0 / (1.0 + exp(-1.1));

  Matrix* output = feedforward(nn, input);
  CU_ASSERT_PTR_NOT_NULL(output);
  CU_ASSERT_TRUE(compare_matrices(output, expected_output, 1e-9));

  free_matrix(input);
  free_matrix(output);
  free_matrix(expected_output);
  free_network(nn);
}

/**
 * @brief Tests the backpropagation algorithm with Softmax activation and
 * Categorical Cross-Entropy loss. Verifies that the delta calculated during
 * backpropagation is correct for this specific combination.
 */
void test_backpropagate_softmax_cce() {
  NeuralNetwork* nn = create_network(1);
  CU_ASSERT_PTR_NOT_NULL(nn);

  // Create a simple layer with Softmax activation
  Layer* layer = (Layer*)malloc(sizeof(Layer));
  CU_ASSERT_PTR_NOT_NULL(layer);
  layer->weights = create_matrix(2, 2);  // 2 inputs, 2 outputs
  fill_matrix(layer->weights, 0.5);
  layer->bias = create_matrix(1, 2);
  fill_matrix(layer->bias, 0.1);
  layer->activation_type = SOFTMAX;
  layer->leak_parameter = 0.0;
  nn->layers[0] = layer;

  // Input matrix
  Matrix* input = create_matrix(1, 2);  // 1 sample, 2 features
  input->matrix_data[0] = 1.0;
  input->matrix_data[1] = 1.0;

  // True labels
  Matrix* y_true = create_matrix(1, 2);
  y_true->matrix_data[0] = 0.0;
  y_true->matrix_data[1] = 1.0;

  // Run feedforward to populate cache
  Matrix* output = feedforward(nn, input);
  CU_ASSERT_PTR_NOT_NULL(output);

  // Expected delta for softmax + CCE: y_hat - y_true
  Matrix* expected_delta = subtract_matrix(output, y_true);

  // Run backpropagate
  backpropagate(
      nn, y_true, CCE,
      categorical_cross_entropy_gradient);  // CCE is the LossFunctionType

  // Retrieve delta from cache
  char delta_key[32];
  sprintf(delta_key, "delta_%zu", (size_t)0);
  Matrix* actual_delta = cache_get(nn->cache, delta_key);

  CU_ASSERT_PTR_NOT_NULL(actual_delta);
  CU_ASSERT_TRUE(compare_matrices(actual_delta, expected_delta, 1e-9));

  free_matrix(input);
  free_matrix(y_true);
  free_matrix(output);
  free_matrix(expected_delta);
  free_matrix(actual_delta);
  free_network(nn);
}

/**
 * @brief Array of CU_TestInfo structures for neural network tests.
 */
CU_TestInfo nn_tests[] = {
    {"test_sigmoid", test_sigmoid},
    {"test_create_free_network", test_create_free_network},
    {"test_feedforward_simple", test_feedforward_simple},
    {"test_backpropagate_softmax_cce", test_backpropagate_softmax_cce},
    CU_TEST_INFO_NULL};
