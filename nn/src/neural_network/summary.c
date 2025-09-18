#include <math.h>
#include <stdio.h>

#include "feedforward.h"
#include "neural_network.h"
#include "utils.h"

void fprint_network_summary(FILE* stream, const NeuralNetwork* nn) {
  if (nn == NULL) {
    fprintf(stream, "Neural network is NULL.\n");
    return;
  }

  fprintf(stream, "==================================\n");
  fprintf(stream, "      Neural Network Summary      \n");
  fprintf(stream, "==================================\n");
  fprintf(stream, "Number of layers: %zu\n", nn->num_layers);

  for (size_t i = 0; i < nn->num_layers; i++) {
    Layer* layer = nn->layers[i];
    fprintf(stream, "----------------------------------\n");
    fprintf(stream, "Layer %zu:\n", i + 1);
    fprintf(stream, "  Weights matrix: %zu x %zu\n", layer->weights->rows,
            layer->weights->cols);
    fprintf(stream, "  Bias matrix:    %zu x %zu\n", layer->bias->rows,
            layer->bias->cols);
    fprintf(stream, "  Activation:     %s\n",
            activation_to_string(layer->activation_type));
  }
  fprintf(stream, "==================================\n");
}

void flog_training_progress(FILE* stream, int epoch, int epochs, double loss) {
  if (epoch % 100 == 0 || epoch == epochs - 1) {
    fprintf(stream, "Epoch %d/%d, Loss: %f\n", epoch, epochs, loss);
  }
}

void fprint_model_predictions(FILE* stream, const NeuralNetwork* nn,
                              const Matrix* x_test, const Matrix* y_test) {
  if (nn == NULL || x_test == NULL || y_test == NULL) {
    fprintf(stream,
            "Cannot print predictions: NeuralNetwork, x_test, or y_test is "
            "NULL.\n");
    return;
  }

  Matrix* predictions = feedforward(nn, x_test);
  if (predictions == NULL) {
    fprintf(stream, "Failed to generate predictions.\n");
    return;
  }

  fprintf(stream, "\n==================================\n");
  fprintf(stream, "         Model Predictions        \n");
  fprintf(stream, "==================================\n");
  fprintf(stream, "Input -> Expected | Predicted (Rounded)\n");
  fprintf(stream, "----------------------------------\n");

  for (size_t i = 0; i < x_test->rows; i++) {
    fprintf(stream, "Input: (");
    for (size_t j = 0; j < x_test->cols; j++) {
      fprintf(stream, "%.0f%s", x_test->matrix_data[i * x_test->cols + j],
              (j == x_test->cols - 1) ? "" : ", ");
    }
    fprintf(stream, ") -> Expected: %.0f | Predicted: %.4f (Rounded: %.0f)\n",
            y_test->matrix_data[i], predictions->matrix_data[i],
            round(predictions->matrix_data[i]));
  }
  fprintf(stream, "==================================\n");

  free_matrix(predictions);
}