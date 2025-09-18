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

/**
 * @brief Reads a matrix from a text file.
 * The file format is expected to be: first line for rows, second for columns,
 * then matrix data.
 * @param filename The path to the file to read.
 * @return A pointer to the newly created Matrix, or NULL if an error occurs.
 */
Matrix* read_matrix(const char* filename) {
  LOG_INFO("Attempting to load matrix from file: %s", filename);

  FILE* file = fopen(filename, "r");
  ASSERT(file != NULL, "Failed to open file for matrix loading.");

  char entry[1024];
  size_t rows = 0, cols = 0;

  char* endptr;
  Matrix* m = NULL;  // Initialize m to NULL

  if (fgets(entry, sizeof(entry), file) == NULL) {
    LOG_ERROR("Could not read rows from file: %s", filename);
    fclose(file);
    return NULL;
  }
  rows = strtol(entry, &endptr, 10);
  if (endptr == entry ||
      (*endptr != '\n' && *endptr != '\0')) {  // Added parentheses
    LOG_ERROR("Invalid row format in file: %s", filename);
    fclose(file);
    return NULL;
  }

  if (fgets(entry, sizeof(entry), file) == NULL) {
    LOG_ERROR("Could not read columns from file: %s", filename);
    fclose(file);
    return NULL;
  }
  cols = strtol(entry, &endptr, 10);
  if (endptr == entry ||
      (*endptr != '\n' && *endptr != '\0')) {  // Added parentheses
    LOG_ERROR("Invalid column format in file: %s", filename);
    // No need to free m here, as it's still NULL
    fclose(file);
    return NULL;
  }

  ASSERT(rows > 0 && cols > 0, "Invalid matrix dimensions read from file.");

  m = create_matrix(rows, cols);  // Assign to m after successful creation

  for (size_t i = 0; i < rows; i++) {
    if (fgets(entry, sizeof(entry), file) == NULL) {
      LOG_ERROR("Unexpected end of file while reading matrix data.");
      free_matrix(m);
      fclose(file);
      return NULL;
    }

    char* line_ptr = entry;
    char* prev_line_ptr = entry;
    for (size_t j = 0; j < cols; j++) {
      m->matrix_data[i * cols + j] = strtod(line_ptr, &line_ptr);
      if (line_ptr == prev_line_ptr) {
        LOG_ERROR("Invalid number format in matrix data at row %zu, col %zu.",
                  i, j);
        free_matrix(m);
        fclose(file);
        return NULL;
      }
      prev_line_ptr = line_ptr;
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

/**
 * @brief Creates a deep copy of an existing matrix.
 * @param m A pointer to the source Matrix to be copied.
 * @return A pointer to the newly created deep copy of the matrix.
 */
Matrix* copy_matrix(const Matrix* m) {
  ASSERT(m != NULL, "Input matrix for copy is NULL.");

  LOG_INFO("Copying a %zux%zu matrix.", m->rows, m->cols);
  Matrix* new_matrix = create_matrix(m->rows, m->cols);

  size_t total_bytes = m->rows * m->cols * sizeof(double);
  memcpy(new_matrix->matrix_data, m->matrix_data, total_bytes);

  LOG_INFO("Matrix copy complete.");

  return new_matrix;
}

/**
 * @brief Fills all elements of a matrix with a specified scalar value.
 * @param m A pointer to the Matrix to be filled.
 * @param n The double value to fill the matrix with.
 */
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

/**
 * @brief Randomizes the elements of a matrix within a specific range.
 * The range is determined by `n` (typically the number of input features) to
 * help prevent vanishing/exploding gradients.
 * @param m A pointer to the Matrix to be randomized.
 * @param n A scaling factor used to determine the range of random values.
 */
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

/**
 * @brief Frees the memory allocated for a matrix.
 * @param m A pointer to the Matrix to be freed.
 */
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

/**
 * @brief Prints the elements of a matrix to standard output for debugging
 * purposes.
 * @param m A pointer to the Matrix to be printed.
 */
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

/**
 * @brief Writes a matrix to a text file.
 * The format written is: rows\n, cols\n, then matrix data with space-separated
 * values.
 * @param m A pointer to the Matrix to be written.
 * @param filename The path to the file where the matrix will be saved.
 */
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

/**
 * @brief Finds the index of the maximum element in a flattened matrix.
 * @param m A pointer to the Matrix.
 * @return The 0-based index of the maximum element.
 */
size_t matrix_argmax(Matrix* m) {
  ASSERT(m != NULL, "Input matrix for argmax is NULL.");

  double maxValue = m->matrix_data[0];
  size_t maxIndex = 0;

  size_t total_elements = m->rows * m->cols;
  for (size_t i = 1; i < total_elements; i++) {
    if (m->matrix_data[i] > maxValue) {
      maxIndex = i;
      maxValue = m->matrix_data[i];
    }
  }
  LOG_INFO("Max value found at index %zu.", maxIndex);
  return maxIndex;
}