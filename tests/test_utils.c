/**
 * @file test_utils.c
 * @brief Utility functions for testing purposes.
 */

#include "test_utils.h"

#include <math.h>

/**
 * @brief Compares two matrices for approximate equality.
 * @param m1 Pointer to the first matrix.
 * @param m2 Pointer to the second matrix.
 * @param epsilon The maximum allowed difference between corresponding elements.
 * @return 1 if the matrices are approximately equal, 0 otherwise.
 */
int compare_matrices(Matrix* m1, Matrix* m2, double epsilon) {
  if (m1 == NULL || m2 == NULL) return 0;
  if (m1->rows != m2->rows || m1->cols != m2->cols) return 0;

  size_t total_elements = m1->rows * m1->cols;
  for (size_t i = 0; i < total_elements; i++) {
    if (fabs(m1->matrix_data[i] - m2->matrix_data[i]) > epsilon) {
      return 0;
    }
  }
  return 1;
}
