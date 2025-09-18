#ifndef NN_SUMMARY_H
#define NN_SUMMARY_H

#include <stdio.h>

#include "neural_network.h"

void fprint_network_summary(FILE* stream, const NeuralNetwork* nn);
void flog_training_progress(FILE* stream, int epoch, int epochs, double loss);
void fprint_model_predictions(FILE* stream, const NeuralNetwork* nn,
                              const Matrix* x_test, const Matrix* y_test);

#endif  // NN_SUMMARY_H
