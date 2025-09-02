#pragma once

#include "activation.h"
#include "cache.h"
#include "linalg.h"

//==============================
// Neural Network Layer Struct
//==============================

typedef Matrix* (*ActivationFunc)(Matrix*);

typedef struct {
  Matrix* weights;
  Matrix* bias;
  
  ActivationFunc activation;

  // The leak parameter for the Leaky ReLU activation function.
  double leak_parameter;
} Layer;


//==============================
// Neural Network Struct
//==============================

typedef struct {
  Layer** layers;
  int num_layers;
  
  // A caching mechanism to store intermediate values from the forward pass.
  Cache* cache;
} NeuralNetwork;
