#include <stdio.h>

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