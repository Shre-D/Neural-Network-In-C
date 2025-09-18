
/**
 * @file mnist.c
 * @brief Example demonstrating a neural network solving the MNIST handwritten
 * digit recognition problem. This program trains a multi-layer perceptron on
 * the MNIST dataset and evaluates its accuracy.
 */

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
#include "utils.h"

/**
 * @brief Helper function to write a matrix to a file in a human-readable
 * format.
 * @param file A pointer to the FILE stream.
 * @param name The name of the matrix (e.g., "Layer 1 Weights").
 * @param m A pointer to the Matrix to write.
 */
void write_matrix_to_file(FILE* file, const char* name, const Matrix* m) {
  fprintf(file, "%s:\n", name);
  fprintf(file, "Rows: %zu, Cols: %zu\n", m->rows, m->cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      fprintf(file, "%.6f ", m->matrix_data[i * m->cols + j]);
    }
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
}

/**
 * @brief Reads data from a CSV file into a Matrix structure.
 * Assumes the CSV contains numerical data separated by commas.
 * @param filename The path to the CSV file.
 * @param rows The expected number of rows in the matrix.
 * @param cols The expected number of columns in the matrix.
 * @return A pointer to the newly created Matrix containing the CSV data, or
 * NULL on error.
 */
Matrix* read_csv(const char* filename, size_t rows, size_t cols) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    LOG_ERROR("Could not open file %s", filename);
    return NULL;
  }

  Matrix* data = create_matrix(rows, cols);
  if (!data) {
    fclose(file);
    return NULL;
  }

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      if (fscanf(file, "%lf,", &data->matrix_data[i * cols + j]) != 1) {
        LOG_ERROR("Error reading data from %s", filename);
        free_matrix(data);
        fclose(file);
        return NULL;
      }
    }
  }

  fclose(file);
  return data;
}

/**
 * @brief Main function to run the MNIST handwritten digit recognition example.
 * Loads MNIST data, constructs and trains a neural network, evaluates its
 * performance, and logs training progress and model parameters.
 * @return 0 on successful execution, 1 on error.
 */
int main() {
  srand(time(NULL));

  NeuralNetwork* nn;

  // Open log files
  FILE* training_log_file = fopen("training_log.txt", "w");
  if (!training_log_file) {
    LOG_ERROR("Could not open training_log.txt");
    return 1;
  }

  FILE* model_summary_file = fopen("model_summary.txt", "w");
  if (!model_summary_file) {
    LOG_ERROR("Could not open model_summary.txt");
    fclose(training_log_file);
    return 1;
  }

  // Log model architecture to summary file
  fprintf(model_summary_file, "\n--- Model Architecture ---\n");
  size_t layers_sizes[] = {784, 128, 10};
  size_t num_layers = sizeof(layers_sizes) / sizeof(layers_sizes[0]) - 1;
  fprintf(model_summary_file, "Input Layer: %zu neurons\n", layers_sizes[0]);
  for (size_t i = 0; i < num_layers; i++) {
    fprintf(model_summary_file,
            "Hidden Layer %zu: %zu neurons, Activation: ", i + 1,
            layers_sizes[i + 1]);
    if (i == num_layers - 1) {
      fprintf(model_summary_file, "Softmax\n");
    } else {
      fprintf(model_summary_file, "ReLU\n");
    }
  }
  fprintf(model_summary_file, "\n");

  // Load the MNIST dataset
  // Using pre-converted CSV files for simplicity
  // The first column is the label, the rest are pixel values
  Matrix* train_data = read_csv("../data/mnist/mnist_train.csv", 60000, 785);
  Matrix* test_data = read_csv("../data/mnist/mnist_test.csv", 10000, 785);

  if (!train_data || !test_data) {
    LOG_ERROR("Failed to load MNIST dataset.");
    fclose(training_log_file);
    fclose(model_summary_file);
    return 1;
  }

  // Separate labels and images
  Matrix* train_images = create_matrix(60000, 784);
  Matrix* train_labels_matrix = create_matrix(60000, 1);
  for (int i = 0; i < 60000; i++) {
    train_labels_matrix->matrix_data[i] = train_data->matrix_data[i * 785];
    memcpy(&train_images->matrix_data[i * 784],
           &train_data->matrix_data[i * 785 + 1], 784 * sizeof(double));
  }

  Matrix* test_images = create_matrix(10000, 784);
  Matrix* test_labels_matrix = create_matrix(10000, 1);
  for (int i = 0; i < 10000; i++) {
    test_labels_matrix->matrix_data[i] = test_data->matrix_data[i * 785];
    memcpy(&test_images->matrix_data[i * 784],
           &test_data->matrix_data[i * 785 + 1], 784 * sizeof(double));
  }

  // Normalize image data
  for (size_t i = 0; i < train_images->rows * train_images->cols; i++) {
    train_images->matrix_data[i] /= 255.0;
  }
  for (size_t i = 0; i < test_images->rows * test_images->cols; i++) {
    test_images->matrix_data[i] /= 255.0;
  }

  // One-hot encode labels
  Matrix* train_labels = create_matrix(60000, 10);
  for (size_t i = 0; i < 60000; i++) {
    int label = (int)train_labels_matrix->matrix_data[i];
    train_labels->matrix_data[i * 10 + label] = 1.0;
  }

  Matrix* test_labels = create_matrix(10000, 10);
  for (size_t i = 0; i < 10000; i++) {
    int label = (int)test_labels_matrix->matrix_data[i];
    test_labels->matrix_data[i * 10 + label] = 1.0;
  }

  // Create the neural network
  nn = create_network(num_layers);

  for (size_t i = 0; i < num_layers; i++) {
    nn->layers[i] = (Layer*)malloc(sizeof(Layer));
    nn->layers[i]->weights =
        create_matrix(layers_sizes[i], layers_sizes[i + 1]);
    nn->layers[i]->bias = create_matrix(1, layers_sizes[i + 1]);
    randomize_matrix(nn->layers[i]->weights, 0.1);
    fill_matrix(nn->layers[i]->bias, 0.0);
    nn->layers[i]->activation_type = (i == num_layers - 1) ? SOFTMAX : RELU;
  }

  // Training parameters
  double learning_rate = 0.01;
  int epochs = 10;
  int batch_size = 32;

  // Training loop
  for (int epoch = 0; epoch < epochs; epoch++) {
    double total_loss = 0;
    for (size_t i = 0; i < train_images->rows; i += batch_size) {
      // Create mini-batch
      size_t current_batch_size = (i + batch_size > train_images->rows)
                                      ? (train_images->rows - i)
                                      : (size_t)batch_size;
      Matrix* batch_images = create_matrix(current_batch_size, 784);
      Matrix* batch_labels = create_matrix(current_batch_size, 10);

      memcpy(batch_images->matrix_data, &train_images->matrix_data[i * 784],
             current_batch_size * 784 * sizeof(double));
      memcpy(batch_labels->matrix_data, &train_labels->matrix_data[i * 10],
             current_batch_size * 10 * sizeof(double));

      // Forward pass
      Matrix* y_hat = feedforward(nn, batch_images);

      // Calculate loss
      total_loss += categorical_cross_entropy(y_hat, batch_labels);

      // Backward pass
      backpropagate(nn, batch_labels, CCE, categorical_cross_entropy_gradient);

      // Update weights and biases
      for (size_t j = 0; j < nn->num_layers; j++) {
        Matrix* dW = calculate_weight_gradient(nn->cache, j, nn->num_layers);
        Matrix* db = calculate_bias_gradient(nn->cache, j, nn->num_layers);

        Matrix* scaled_dW = scale_matrix(learning_rate, dW);
        Matrix* scaled_db = scale_matrix(learning_rate, db);

        Matrix* new_weights =
            subtract_matrix(nn->layers[j]->weights, scaled_dW);
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
      free_matrix(batch_images);
      free_matrix(batch_labels);
    }
    fprintf(training_log_file, "Epoch %d, Loss: %f\n", epoch + 1,
            total_loss / (train_images->rows / batch_size));
  }

  // Evaluate on test set
  Matrix* test_output = feedforward(nn, test_images);
  int correct_predictions = 0;
  for (size_t i = 0; i < test_output->rows; i++) {
    int predicted_label = 0;
    double max_prob = 0;
    for (size_t j = 0; j < 10; j++) {
      if (test_output->matrix_data[i * 10 + j] > max_prob) {
        max_prob = test_output->matrix_data[i * 10 + j];
        predicted_label = j;
      }
    }

    int true_label = 0;
    for (size_t j = 0; j < 10; j++) {
      if (test_labels->matrix_data[i * 10 + j] == 1.0) {
        true_label = j;
        break;
      }
    }

    if (predicted_label == true_label) {
      correct_predictions++;
    }
  }

  double accuracy = (double)correct_predictions / test_output->rows;
  fprintf(training_log_file, "Test Accuracy: %f%%\n", accuracy * 100);

  // Save model summary (weights and biases) to file
  fprintf(model_summary_file, "\n--- Trained Model Parameters ---\n");
  for (size_t i = 0; i < num_layers; i++) {
    char weight_name[50];
    char bias_name[50];
    sprintf(weight_name, "Layer %zu Weights", i + 1);
    sprintf(bias_name, "Layer %zu Bias", i + 1);
    write_matrix_to_file(model_summary_file, weight_name,
                         nn->layers[i]->weights);
    write_matrix_to_file(model_summary_file, bias_name, nn->layers[i]->bias);
  }

  // Free memory
  free_matrix(train_data);
  free_matrix(test_data);
  free_matrix(train_images);
  free_matrix(train_labels_matrix);
  free_matrix(train_labels);
  free_matrix(test_images);
  free_matrix(test_labels_matrix);
  free_matrix(test_labels);
  free_matrix(test_output);
  free_network(nn);

  fclose(training_log_file);
  fclose(model_summary_file);

  return 0;
}
