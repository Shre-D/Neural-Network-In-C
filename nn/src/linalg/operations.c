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

#ifdef USE_OPENMP
#include <omp.h>
#endif

//============================
// Functions for Matrix Operations
//============================

/**
 * @brief Creates an identity matrix of size n x n.
 * @param n The dimension of the square identity matrix.
 * @return A pointer to the newly created identity Matrix.
 */
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

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = a->matrix_data[i] + b->matrix_data[i];
  }

  LOG_INFO("Matrix addition complete.");
  return result;
}

/**
 * @brief Performs element-wise subtraction of two matrices.
 * @param a The first matrix (minuend).
 * @param b The second matrix (subtrahend).
 * @return A new matrix containing the result of a - b.
 */
Matrix* subtract_matrix(Matrix* a, Matrix* b) {
  ASSERT(a != NULL && b != NULL, "Input matrices cannot be NULL.");
  ASSERT(a->rows == b->rows && a->cols == b->cols,
         "Matrices must have the same dimensions for subtraction.");

  LOG_INFO("Subtracting two %zux%zu matrices.", a->rows, a->cols);
  Matrix* result = create_matrix(a->rows, a->cols);
  size_t total_elements = a->rows * a->cols;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = a->matrix_data[i] - b->matrix_data[i];
  }

  LOG_INFO("Matrix subtraction complete.");
  return result;
}

/**
 * @brief Performs element-wise multiplication of two matrices (Hadamard
 * product).
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the element-wise product of a and b.
 */
Matrix* multiply_matrix(Matrix* a, Matrix* b) {
  ASSERT(a != NULL && b != NULL, "Input matrices cannot be NULL.");
  ASSERT(a->rows == b->rows && a->cols == b->cols,
         "Matrices must have the same dimensions for element-wise "
         "multiplication.");

  LOG_INFO("Multiplying two %zux%zu matrices element-wise.", a->rows, a->cols);
  Matrix* result = create_matrix(a->rows, a->cols);
  size_t total_elements = a->rows * a->cols;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = a->matrix_data[i] * b->matrix_data[i];
  }

  LOG_INFO("Element-wise matrix multiplication complete.");
  return result;
}

/**
 * @brief Applies a given function to each element of a matrix.
 * @param func A function pointer that takes a double and returns a double.
 * @param m The input matrix.
 * @return A new matrix with the function applied to each element.
 */
Matrix* apply_onto_matrix(double (*func)(double), Matrix* m) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Applying a function to each element of a %zux%zu matrix.", m->rows,
           m->cols);
  Matrix* result = create_matrix(m->rows, m->cols);
  size_t total_elements = m->rows * m->cols;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = func(m->matrix_data[i]);
  }

  LOG_INFO("Function application to matrix complete.");
  return result;
}

/**
 * @brief Adds a scalar value to each element of a matrix.
 * @param m The input matrix.
 * @param n The scalar value to add.
 * @return A new matrix with the scalar added to each element.
 */
Matrix* add_scalar_to_matrix(Matrix* m, double n) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Adding scalar %.2f to a %zux%zu matrix.", n, m->rows, m->cols);
  Matrix* result = create_matrix(m->rows, m->cols);
  size_t total_elements = m->rows * m->cols;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = m->matrix_data[i] + n;
  }

  LOG_INFO("Scalar addition complete.");
  return result;
}

/**
 * @brief Performs the dot product (matrix multiplication) of two matrices.
 * @param a The first matrix.
 * @param b The second matrix.
 * @return A new matrix containing the result of the dot product a * b.
 */
Matrix* dot_matrix(Matrix* m1, Matrix* m2) {
  ASSERT(m1 != NULL && m2 != NULL, "Input matrices cannot be NULL.");
  ASSERT(m1->cols == m2->rows,
         "Matrices dimensions are incompatible for dot product.");

  LOG_INFO("Performing dot product of %zux%zu and %zux%zu matrices.", m1->rows,
           m1->cols, m2->rows, m2->cols);
  Matrix* result = create_matrix(m1->rows, m2->cols);
  // Initialize result matrix with zeros
  memset(result->matrix_data, 0, m1->rows * m2->cols * sizeof(double));

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t i = 0; i < m1->rows; i++) {
    for (size_t j = 0; j < m2->cols; j++) {
      for (size_t k = 0; k < m1->cols; k++) {
        result->matrix_data[i * result->cols + j] +=
            m1->matrix_data[i * m1->cols + k] *
            m2->matrix_data[k * m2->cols + j];
      }
    }
  }

  LOG_INFO("Dot product complete. Resulting matrix is %zux%zu.", result->rows,
           result->cols);
  return result;
}

/**
 * @brief Transposes a matrix.
 * @param m The input matrix.
 * @return A new matrix that is the transpose of the input matrix.
 */
Matrix* transpose_matrix(Matrix* m) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Transposing a %zux%zu matrix.", m->rows, m->cols);
  Matrix* result = create_matrix(m->cols, m->rows);

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
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

/**
 * @brief Scales all elements of a matrix by a scalar value.
 * @param n The scalar value to multiply by.
 * @param m The input matrix.
 * @return A new matrix with all elements scaled by n.
 */
Matrix* scale_matrix(double n, Matrix* m) {
  ASSERT(m != NULL, "Input matrix cannot be NULL.");
  LOG_INFO("Scaling a %zux%zu matrix by %.2f.", m->rows, m->cols, n);
  Matrix* result = create_matrix(m->rows, m->cols);
  size_t total_elements = m->rows * m->cols;

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < total_elements; i++) {
    result->matrix_data[i] = m->matrix_data[i] * n;
  }

  LOG_INFO("Matrix scaling complete.");
  return result;
}

/**
 * @brief Adds a bias vector to each row of a matrix.
 * @param m The input matrix.
 * @param bias The bias vector (must be a 1xN row vector where N is m->cols).
 * @return A new matrix with the bias added to each row.
 */
Matrix* add_bias_to_matrix(Matrix* m, Matrix* bias) {
  ASSERT(m != NULL, "Input matrix is NULL.");
  ASSERT(bias != NULL, "Bias matrix is NULL.");
  ASSERT(bias->rows == 1, "Bias must be a row vector.");
  ASSERT(m->cols == bias->cols,
         "Matrix and bias dimensions are incompatible for addition.");

  Matrix* result = create_matrix(m->rows, m->cols);
  ASSERT(result != NULL, "Failed to create matrix for bias addition.");

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++) {
      result->matrix_data[i * m->cols + j] =
          m->matrix_data[i * m->cols + j] + bias->matrix_data[j];
    }
  }

  return result;
}

/**
 * @brief Sums the columns of a matrix, returning a row vector.
 * @param m The input matrix.
 * @return A new 1xN matrix (row vector) where each element is the sum of the
 * corresponding column in m.
 */
Matrix* sum_matrix_columns(Matrix* m) {
  ASSERT(m != NULL, "Input matrix is NULL.");

  Matrix* result = create_matrix(1, m->cols);
  ASSERT(result != NULL, "Failed to create matrix for column summation.");

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
  for (size_t j = 0; j < m->cols; j++) {
    double sum = 0;
    for (size_t i = 0; i < m->rows; i++) {
      sum += m->matrix_data[i * m->cols + j];
    }
    result->matrix_data[j] = sum;
  }

  return result;
}
