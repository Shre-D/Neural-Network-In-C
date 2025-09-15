/**
 * @file operations.c
 * @brief Linear algebra operations over the Matrix type.
 *
 * Provides identity, elementwise arithmetic, apply, dot product, transpose,
 * and scaling. Functions return newly allocated matrices; callers must free
 * results with free_matrix.
 */
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"
#include "utils.h"

//============================
// Functions for Matrix Operations
//============================

Matrix* identity_matrix(size_t n) {
  LOG_INFO("Creating a %zux%zu identity matrix.", n, n);
  ASSERT(n > 0, "Matrix size must be greater than 0.");
  Matrix* m = create_matrix(n, n);
  // Use memset to efficiently initialize all elements to 0
  memset(m->matrix_data, 0, n * n * sizeof(double));
  for (size_t i = 0; i < n; i++) {
    m->matrix_data[i * n + i] = 1.0;
  }
  LOG_INFO("Identity matrix created successfully.");
  return m;
}

Matrix* add_matrix(Matrix* a, Matrix* b) {
  ASSERT(a != NULL && b != NULL, "Input matrices cannot be NULL.");
  ASSERT(a->rows == b->rows && a->cols == b->cols,
         "Matrices must have the same dimensions for addition.");

  LOG_INFO("Adding two %zux%zu matrices.", a->rows, a->cols);
  Matrix* result = create_matrix(a->rows, a->cols);
  size_t total_elements = a->rows * a->cols;

  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = a->matrix_data[i] + b->matrix_data[i];
  }

  LOG_INFO("Matrix addition complete.");
  return result;
}

Matrix* subtract_matrix(Matrix* a, Matrix* b) {
  ASSERT(a != NULL && b != NULL, "Input matrices cannot be NULL.");
  ASSERT(a->rows == b->rows && a->cols == b->cols,
         "Matrices must have the same dimensions for subtraction.");

  LOG_INFO("Subtracting two %zux%zu matrices.", a->rows, a->cols);
  Matrix* result = create_matrix(a->rows, a->cols);
  size_t total_elements = a->rows * a->cols;

  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = a->matrix_data[i] - b->matrix_data[i];
  }

  LOG_INFO("Matrix subtraction complete.");
  return result;
}

Matrix* multiply_matrix(Matrix* a, Matrix* b) {
  ASSERT(a != NULL && b != NULL, "Input matrices cannot be NULL.");
  ASSERT(a->rows == b->rows && a->cols == b->cols,
         "Matrices must have the same dimensions for element-wise "
         "multiplication.");

  LOG_INFO("Multiplying two %zux%zu matrices element-wise.", a->rows, a->cols);
  Matrix* result = create_matrix(a->rows, a->cols);
  size_t total_elements = a->rows * a->cols;

  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = a->matrix_data[i] * b->matrix_data[i];
  }

  LOG_INFO("Element-wise matrix multiplication complete.");
  return result;
}

Matrix* apply_onto_matrix(double (*func)(double), Matrix* m) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Applying a function to each element of a %zux%zu matrix.", m->rows,
           m->cols);
  Matrix* result = create_matrix(m->rows, m->cols);
  size_t total_elements = m->rows * m->cols;

  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = func(m->matrix_data[i]);
  }

  LOG_INFO("Function application to matrix complete.");
  return result;
}

Matrix* add_scalar_to_matrix(Matrix* m, double n) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Adding scalar %.2f to a %zux%zu matrix.", n, m->rows, m->cols);
  Matrix* result = create_matrix(m->rows, m->cols);
  size_t total_elements = m->rows * m->cols;

  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = m->matrix_data[i] + n;
  }

  LOG_INFO("Scalar addition complete.");
  return result;
}

Matrix* dot_matrix(Matrix* a, Matrix* b) {
  ASSERT(a != NULL && b != NULL, "Input matrices cannot be NULL.");
  ASSERT(a->cols == b->rows,
         "The number of columns in the first matrix must equal the number of "
         "rows in the second matrix for dot product.");

  LOG_INFO("Performing dot product on a %zux%zu and a %zux%zu matrix.", a->rows,
           a->cols, b->rows, b->cols);
  Matrix* result = create_matrix(a->rows, b->cols);

  for (size_t i = 0; i < a->rows; i++) {
    for (size_t j = 0; j < b->cols; j++) {
      double sum = 0;
      for (size_t k = 0; k < a->cols; k++) {
        sum +=
            a->matrix_data[i * a->cols + k] * b->matrix_data[k * b->cols + j];
      }
      result->matrix_data[i * result->cols + j] = sum;
    }
  }

  LOG_INFO("Matrix dot product complete. Resulting matrix is %zux%zu.",
           result->rows, result->cols);
  return result;
}

Matrix* transpose_matrix(Matrix* m) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Transposing a %zux%zu matrix.", m->rows, m->cols);
  Matrix* result = create_matrix(m->cols, m->rows);

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      result->matrix_data[j * result->cols + i] =
          m->matrix_data[i * m->cols + j];
    }
  }

  LOG_INFO("Matrix transpose complete. Resulting matrix is %zux%zu.",
           result->rows, result->cols);
  return result;
}

Matrix* scale_matrix(double n, Matrix* m) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Scaling a %zux%zu matrix by %.2f.", m->rows, m->cols, n);
  Matrix* result = create_matrix(m->rows, m->cols);
  size_t total_elements = m->rows * m->cols;

  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = m->matrix_data[i] * n;
  }

  LOG_INFO("Matrix scaling complete.");
  return result;
}
