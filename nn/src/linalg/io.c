/**
 * @file io.c
 * @brief Matrix input/output utilities and helpers.
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"
#include "utils.h"

//============================
// Functions for Matrix IO
//============================

Matrix* read_matrix(const char* filename) {
  LOG_INFO("Attempting to load matrix from file: %s", filename);

  FILE* file = fopen(filename, "r");
  ASSERT(file != NULL, "Failed to open file for matrix loading.");

  char entry[1024];
  size_t rows = 0, cols = 0;

  if (fgets(entry, sizeof(entry), file) == NULL) {
    LOG_ERROR("Could not read rows from file: %s", filename);
    fclose(file);
    return NULL;
  }
  rows = atoi(entry);

  if (fgets(entry, sizeof(entry), file) == NULL) {
    LOG_ERROR("Could not read columns from file: %s", filename);
    fclose(file);
    return NULL;
  }
  cols = atoi(entry);

  ASSERT(rows > 0 && cols > 0, "Invalid matrix dimensions read from file.");

  Matrix* m = create_matrix(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    if (fgets(entry, sizeof(entry), file) == NULL) {
      LOG_ERROR("Unexpected end of file while reading matrix data.");
      free_matrix(m);
      fclose(file);
      return NULL;
    }

    char* line_ptr = entry;
    for (size_t j = 0; j < cols; j++) {
      m->matrix_data[i * cols + j] = strtod(line_ptr, &line_ptr);
    }
  }

  LOG_INFO("Successfully loaded a %zux%zu matrix from %s.", m->rows, m->cols,
           filename);
  fclose(file);
  return m;
}

Matrix* create_matrix(size_t rows, size_t cols) {
  LOG_INFO("Creating a new matrix of size %zux%zu.", rows, cols);

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

  LOG_INFO("Copying a %zux%zu matrix.", m->rows, m->cols);
  Matrix* new_matrix = create_matrix(m->rows, m->cols);

  size_t total_bytes = m->rows * m->cols * sizeof(double);
  memcpy(new_matrix->matrix_data, m->matrix_data, total_bytes);

  LOG_INFO("Matrix copy complete.");

  return new_matrix;
}

// Helper function for flattening
Matrix* flatten_column_wise(const Matrix* m) {
  Matrix* new_matrix = create_matrix(m->rows * m->cols, 1);

  size_t k = 0;
  for (size_t j = 0; j < m->cols; j++) {
    for (size_t i = 0; i < m->rows; i++) {
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

  if (axis == 0) {
    LOG_INFO(
        "Flattening matrix row-wise. No operation needed as data is "
        "already "
        "contiguous.");
    m->cols = m->rows * m->cols;
    m->rows = 1;
    return m;
  } else {
    LOG_INFO("Flattening matrix column-wise. A new matrix will be created.");
    return flatten_column_wise(m);
  }
}

void fill_matrix(Matrix* m, double n) {
  ASSERT(m != NULL, "Input matrix for fill_matrix is NULL.");
  LOG_INFO("Filling a %zux%zu matrix with the value %.2f.", m->rows, m->cols,
           n);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      m->matrix_data[i * m->cols + j] = n;
    }
  }
}

void randomize_matrix(Matrix* m, double n) {
  LOG_INFO("Randomizing a %zux%zu matrix.", m->rows, m->cols);
  // Apparently a 1/n or 1/n^2 scaling leads to a vanishing gradient problem
  double min = -1.0 / sqrt(n);
  double max = 1.0 / sqrt(n);
  double range = max - min;

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      double random_value = (double)rand() / (double)RAND_MAX;
      m->matrix_data[i * m->cols + j] = min + random_value * range;
    }
  }
  LOG_INFO("Matrix randomized successfully.");
}

void free_matrix(Matrix* m) {
  LOG_INFO("Freeing matrix at address %p.", m);
  if (m == NULL) {
    LOG_WARN("Attempted to free a NULL pointer.");
    return;
  }
  if (m->matrix_data != NULL) {
    free(m->matrix_data);
  }
  free(m);
  LOG_INFO("Matrix freed successfully.");
}

void print_matrix(Matrix* m) {
  ASSERT(m != NULL, "Input matrix for print is NULL.");
  LOG_INFO("Printing matrix of size %zux%zu.", m->rows, m->cols);
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      printf("%.3f ", m->matrix_data[i * m->cols + j]);
    }
    printf("\n");
  }
}

void write_matrix(Matrix* m, const char* filename) {
  ASSERT(m != NULL, "Input matrix for save is NULL.");
  LOG_INFO("Saving a %zux%zu matrix to file: %s", m->rows, m->cols, filename);

  FILE* file = fopen(filename, "w");
  ASSERT(file != NULL, "Failed to open file for saving matrix.");

  fprintf(file, "%zu\n", m->rows);
  fprintf(file, "%zu\n", m->cols);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      fprintf(file, "%.3f ", m->matrix_data[i * m->cols + j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);
  LOG_INFO("Matrix saved successfully.");
}

int matrix_argmax(Matrix* m) {
  ASSERT(m != NULL, "Input matrix for argmax is NULL.");
  ASSERT(m->cols == 1, "Input must be a column vector (Mx1).");

  double maxValue = INT_MIN;
  int maxIndex = 0;

  for (size_t i = 0; i < m->rows; i++) {
    if (m->matrix_data[i] > maxValue) {
      maxIndex = (int)i;
      maxValue = m->matrix_data[i];
    }
  }
  LOG_INFO("Max value found at index %d.", maxIndex);
  return maxIndex;
}
