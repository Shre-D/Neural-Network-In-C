#pragma once

/**
 * @file linalg.h
 * @brief Matrix data structure and linear algebra primitives.
 *
 * This module defines the `Matrix` type and provides creation, IO, and
 * fundamental operations such as elementwise arithmetic, transpose, dot
 * product, and scaling. Functions generally allocate new matrices for their
 * results; callers own returned matrices and must free them with
 * `free_matrix`.
 */

/**
 * @brief Dense 2D matrix with row/column dimensions.
 *
 * Backed by a contiguous 1D buffer of doubles in row-major order.
 */
// _matrix is used to avoid confusion if the struct is self referencing
// 1D array used, this is way more performant than a 2D array is
typedef struct _Matrix {
  double* matrix_data;
  int rows;
  int cols;
} Matrix;

//==============================
// Public API
//==============================

//=====================
// Matrix IO
//=====================

/** @brief Read a matrix from a text file. */
Matrix* read_matrix(const char* filename);
/** @brief Create an uninitialized matrix with given shape. */
Matrix* create_matrix(int rows, int cols);
/** @brief Deep copy a matrix. */
Matrix* copy_matrix(const Matrix* m);
/** @brief Flatten a matrix along an axis (implementation-specific). */
Matrix* flatten_matrix(Matrix* m, int axis);
/** @brief Fill all elements with a constant value. */
void fill_matrix(Matrix* m, double n);
/** @brief Fill with random values in an implementation-defined range. */
void randomize_matrix(Matrix* m, double n);
/** @brief Free a matrix and its data buffer. */
void free_matrix(Matrix* m);
/** @brief Write a matrix to a text file. */
void write_matrix(Matrix* m, const char* filename);
/** @brief Print a matrix to stdout (for debugging). */
void print_matrix(Matrix* m);
/** @brief Return the index of the maximum element (flattened argmax). */
int matrix_argmax(Matrix* m);

//============================
// Matrix Operations
//============================
/** @brief Create an identity matrix of size n×n. */
Matrix* identity_matrix(int n);
/** @brief Elementwise addition: result = a + b. */
Matrix* add_matrix(Matrix* a, Matrix* b);
/** @brief Elementwise subtraction: result = a - b. */
Matrix* subtract_matrix(Matrix* a, Matrix* b);
/** @brief Elementwise multiplication: result = a ⊙ b. */
Matrix* multiply_matrix(Matrix* a, Matrix* b);
/** @brief Apply scalar function to each element: result[i] = func(m[i]). */
Matrix* apply_onto_matrix(double* (*func)(double), Matrix* m);
/** @brief Add scalar to all elements. */
Matrix* add_scalar_to_matrix(Matrix* m, double n);
/** @brief Matrix product (dot): result = a · b. */
Matrix* dot_matrix(Matrix* a, Matrix* b);
/** @brief Transpose a matrix. */
Matrix* transpose_matrix(Matrix* m);
/** @brief Scale all elements by scalar n. */
Matrix* scale_matrix(double n, Matrix* m);
