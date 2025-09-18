/**
 * @file xor.c
 * @brief Example demonstrating a neural network solving the XOR problem.
 * This program trains a simple feedforward neural network to learn the XOR
 * logic gate.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "activation.h"
#include "backprop.h"
#include "feedforward.h"
#include "linalg.h"
#include "loss.h"
#include "neural_network.h"
#include "summary.h"
#include "utils.h"

/**
 * @brief Main function to run the XOR neural network example.
 * Initializes training data, constructs a neural network, trains it using
 * backpropagation, and then tests its performance on the XOR inputs.
 * @return 0 on successful execution, 1 on error.
 */
int main() {
  srand(time(NULL));

  // 1. Define XOR training data
  Matrix* x_train = create_matrix(4, 2);
  x_train->matrix_data[0] = 0.0;
  x_train->matrix_data[1] = 0.0;
  x_train->matrix_data[2] = 0.0;
  x_train->matrix_data[3] = 1.0;
  x_train->matrix_data[4] = 1.0;
  x_train->matrix_data[5] = 0.0;
  x_train->matrix_data[6] = 1.0;
  x_train->matrix_data[7] = 1.0;

  Matrix* y_train = create_matrix(4, 1);
  y_train->matrix_data[0] = 0.0;
  y_train->matrix_data[1] = 1.0;
  y_train->matrix_data[2] = 1.0;
  y_train->matrix_data[3] = 0.0;

  // 2. Network Architecture
  size_t layers_sizes[] = {2, 4, 1};
  size_t num_layers = sizeof(layers_sizes) / sizeof(layers_sizes[0]) - 1;

  NeuralNetwork* nn = create_network(num_layers);
  if (nn == NULL) {
    LOG_ERROR("Failed to create neural network.");
    return 1;
  }

  for (size_t i = 0; i < num_layers; i++) {
    nn->layers[i] = (Layer*)malloc(sizeof(Layer));
    if (nn->layers[i] == NULL) {
      LOG_ERROR("Failed to allocate memory for layer %zu.", i);
      free_network(nn);
      free_matrix(x_train);
      free_matrix(y_train);
      return 1;
    }
    nn->layers[i]->weights =
        create_matrix(layers_sizes[i], layers_sizes[i + 1]);
    nn->layers[i]->bias = create_matrix(1, layers_sizes[i + 1]);

    randomize_matrix(nn->layers[i]->weights, 0.1);
    fill_matrix(nn->layers[i]->bias, 0.0);

    nn->layers[i]->activation_type = (i == num_layers - 1) ? SIGMOID : RELU;
    nn->layers[i]->leak_parameter = 0.01;
  }

  // 3. Print Network Summary
  FILE* summary_file = fopen("model_summary.txt", "w");
  if (summary_file == NULL) {
    LOG_ERROR("Failed to open model_summary.txt for writing.");
    // Handle error, but continue for now to avoid stopping the whole process
  } else {
    fprint_network_summary(summary_file, nn);
    fprint_model_predictions(summary_file, nn, x_train,
                             y_train);  // Add predictions
    fclose(summary_file);
  }

  // 4. Training Parameters
  double learning_rate = 0.1;
  int epochs = 2000;

  printf("Training XOR network with %d epochs, learning rate %.2f\n", epochs,
         learning_rate);

  FILE* log_file = fopen("training_log.txt", "w");
  if (log_file == NULL) {
    LOG_ERROR("Failed to open training_log.txt for writing.");
  }

  // 5. Training Loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_loss = 0;

    Matrix* y_hat = feedforward(nn, x_train);
    if (y_hat == NULL) {
      LOG_ERROR("Feedforward failed at epoch %d.", epoch);
      free_network(nn);
      free_matrix(x_train);
      free_matrix(y_train);
      return 1;
    }

    total_loss = mean_squared_error(y_hat, y_train);

    backpropagate(nn, y_train, MSE, mean_squared_error_gradient);

    for (size_t j = 0; j < nn->num_layers; j++) {
      Matrix* dW = calculate_weight_gradient(nn->cache, j, nn->num_layers);
      Matrix* db = calculate_bias_gradient(nn->cache, j, nn->num_layers);

      Matrix* scaled_dW = scale_matrix(learning_rate, dW);
      Matrix* scaled_db = scale_matrix(learning_rate, db);

      Matrix* new_weights = subtract_matrix(nn->layers[j]->weights, scaled_dW);
      Matrix* new_bias = subtract_matrix(nn->layers[j]->bias, scaled_db);

      free_matrix(nn->layers[j]->weights);
      free_matrix(nn->layers[j]->bias);
      nn->layers[j]->weights = new_weights;
      nn->layers[j]->bias = new_bias;

      free_matrix(dW);
      free_matrix(db);
      free_matrix(scaled_dW);
      free_matrix(scaled_db);
    }
    free_matrix(y_hat);

    if (log_file != NULL) {
      flog_training_progress(log_file, epoch, epochs, total_loss);
    }
  }

  printf("\nTraining complete. Testing network...\n");

  // 6. Testing
  Matrix* predictions = feedforward(nn, x_train);
  if (predictions == NULL) {
    LOG_ERROR("Feedforward failed during testing.");
    free_network(nn);
    free_matrix(x_train);
    free_matrix(y_train);
    return 1;
  }

  printf("XOR Test Results:\n");
  for (size_t i = 0; i < x_train->rows; i++) {
    printf(
        "Input: (%.0f, %.0f) -> Expected: %.0f, Predicted: %.4f (Rounded: "
        "%.0f)\n",
        x_train->matrix_data[i * 2], x_train->matrix_data[i * 2 + 1],
        y_train->matrix_data[i], predictions->matrix_data[i],
        round(predictions->matrix_data[i]));
  }

  // 7. Memory Cleanup
  free_matrix(x_train);
  free_matrix(y_train);
  free_matrix(predictions);

  if (log_file != NULL) fclose(log_file);

  free_network(nn);

  printf("\nMemory freed. XOR example finished.\n");

  return 0;
}
