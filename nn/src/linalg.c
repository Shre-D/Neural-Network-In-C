#include "linalg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

Matrix* create_matrix(int rows, int cols) {
  LOG_INFO("Creating a new matrix of size %dx%d.", rows, cols);

  Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
  CHECK_MALLOC(matrix, "Failed to allocate memory for Matrix struct.");

  matrix->matrix_data = (double*)malloc(rows * cols * sizeof(double));
  CHECK_MALLOC(matrix->matrix_data,
               "Failed to allocate memory for matrix data.");

  matrix->rows = rows;
  matrix->cols = cols;

  LOG_INFO("Matrix created successfully at address %p.", matrix);

  return matrix;
}

Matrix* copy_matrix(const Matrix* m) {
  ASSERT(m != NULL, "Input matrix for copy is NULL.");

  LOG_INFO("Copying a %dx%d matrix.", m->rows, m->cols);
  Matrix* new_matrix = create_matrix(m->rows, m->cols);

  size_t total_bytes = m->rows * m->cols * sizeof(double);
  memcpy(new_matrix->matrix_data, m->matrix_data, total_bytes);

  LOG_INFO("Matrix copy complete.", );

  return new_matrix;
}

// Helper function for flattening
Matrix* flatten_column_wise(const Matrix* m) {
  Matrix* new_matrix = create_matrix(m->rows * m->cols, 1);

  int k = 0;
  for (int j = 0; j < m->cols; j++) {
    for (int i = 0; i < m->rows; i++) {
      new_matrix->matrix_data[k] = m->matrix_data[i * m->cols + j];
      k++;
    }
  }
  return new_matrix;
}

Matrix* flatten_matrix(Matrix* m, int axis) {
  ASSERT(m != NULL, "Input matrix to flatten is NULL.");
  ASSERT(axis == 0 || axis == 1,
         "Axis must be 0 (row-wise) or 1 (column-wise).");

  if (axis == 0) {  // row wise flattening
    LOG_INFO(
        "Flattening matrix row-wise. No operation needed as data is already "
        "contiguous.");
    m->cols = m->rows * m->cols;
    m->rows = 1;
    return m;
  } else {  // axis == 1, column wise flattening, invokes helper written above
    LOG_INFO("Flattening matrix column-wise. A new matrix will be created.");
    return flatten_column_wise(m);
  }
}

void fill_matrix(Matrix* m, double n) {
  ASSERT(m != NULL, "Input matrix for fill_matrix is NULL.");
  LOG_INFO("Filling a %dx%d matrix with the value %.2f.", m->rows, m->cols, n);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->matrix_data[i * m->cols + j] = n;
    }
  }
}
