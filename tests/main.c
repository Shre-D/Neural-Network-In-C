/**
 * @file main.c
 * @brief Main test file for various neural network components.
 * This file contains individual test functions and a main runner to execute
 * them.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "activation.h"
#include "cache.h"
#include "linalg.h"
#include "neural_network.h"
#include "utils.h"

#define TEST_ASSERT(condition, message)      \
  do {                                       \
    if (!(condition)) {                      \
      LOG_ERROR("TEST FAILED: %s", message); \
      return 0;                              \
    }                                        \
  } while (0)

/**
 * @brief Macro to run a single test function.
 * Executes the given test function and reports its success or failure.
 * @param test_func The name of the test function to run.
 */
#define RUN_TEST(test_func)                 \
  do {                                      \
    printf("Running %s...\n", #test_func);  \
    if (test_func()) {                      \
      printf("  %s: PASSED\n", #test_func); \
      tests_passed++;                       \
    } else {                                \
      printf("  %s: FAILED\n", #test_func); \
      tests_failed++;                       \
    }                                       \
    tests_run++;                            \
  } while (0)

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/**
 * @brief Helper function to compare two matrices for approximate equality.
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

/**
 * @brief Tests the creation and freeing of a matrix.
 * Verifies that `create_matrix` allocates memory correctly and `free_matrix`
 * deallocates it without issues.
 * @return 1 on success, 0 on failure.
 */
int test_create_free_matrix() {
  Matrix* m = create_matrix(2, 3);
  TEST_ASSERT(m != NULL, "create_matrix returned NULL");
  TEST_ASSERT(m->rows == 2 && m->cols == 3,
              "create_matrix dimensions incorrect");
  free_matrix(m);
  return 1;
}

/**
 * @brief Tests the element-wise addition of two matrices.
 * Verifies that `add_matrix` correctly sums corresponding elements of two
 * matrices.
 * @return 1 on success, 0 on failure.
 */
int test_add_matrix() {
  Matrix* a = create_matrix(2, 2);
  Matrix* b = create_matrix(2, 2);
  Matrix* expected = create_matrix(2, 2);

  a->matrix_data[0] = 1;
  a->matrix_data[1] = 2;
  a->matrix_data[2] = 3;
  a->matrix_data[3] = 4;

  b->matrix_data[0] = 5;
  b->matrix_data[1] = 6;
  b->matrix_data[2] = 7;
  b->matrix_data[3] = 8;

  expected->matrix_data[0] = 6;
  expected->matrix_data[1] = 8;
  expected->matrix_data[2] = 10;
  expected->matrix_data[3] = 12;

  Matrix* result = add_matrix(a, b);
  TEST_ASSERT(compare_matrices(result, expected, 1e-9), "add_matrix failed");

  free_matrix(a);
  free_matrix(b);
  free_matrix(expected);
  free_matrix(result);
  return 1;
}

/**
 * @brief Tests the dot product (matrix multiplication) of two matrices.
 * Verifies that `dot_matrix` correctly computes the matrix product.
 * @return 1 on success, 0 on failure.
 */
int test_dot_matrix() {
  Matrix* a = create_matrix(2, 2);
  Matrix* b = create_matrix(2, 2);
  Matrix* expected = create_matrix(2, 2);

  a->matrix_data[0] = 1;
  a->matrix_data[1] = 2;
  a->matrix_data[2] = 3;
  a->matrix_data[3] = 4;

  b->matrix_data[0] = 5;
  b->matrix_data[1] = 6;
  b->matrix_data[2] = 7;
  b->matrix_data[3] = 8;

  expected->matrix_data[0] = 19;
  expected->matrix_data[1] = 22;
  expected->matrix_data[2] = 43;
  expected->matrix_data[3] = 50;

  Matrix* result = dot_matrix(a, b);
  TEST_ASSERT(compare_matrices(result, expected, 1e-9), "dot_matrix failed");

  free_matrix(a);
  free_matrix(b);
  free_matrix(expected);
  free_matrix(result);
  return 1;
}

/**
 * @brief Tests the transposition of a matrix.
 * Verifies that `transpose_matrix` correctly computes the transpose of a given
 * matrix.
 * @return 1 on success, 0 on failure.
 */
int test_transpose_matrix() {
  Matrix* m = create_matrix(2, 3);
  Matrix* expected = create_matrix(3, 2);

  m->matrix_data[0] = 1;
  m->matrix_data[1] = 2;
  m->matrix_data[2] = 3;
  m->matrix_data[3] = 4;
  m->matrix_data[4] = 5;
  m->matrix_data[5] = 6;

  expected->matrix_data[0] = 1;
  expected->matrix_data[1] = 4;
  expected->matrix_data[2] = 2;
  expected->matrix_data[3] = 5;
  expected->matrix_data[4] = 3;
  expected->matrix_data[5] = 6;

  Matrix* result = transpose_matrix(m);
  TEST_ASSERT(compare_matrices(result, expected, 1e-9),
              "transpose_matrix failed");

  free_matrix(m);
  free_matrix(expected);
  free_matrix(result);
  return 1;
}

/**
 * @brief Tests the matrix_argmax function.
 * Verifies that `matrix_argmax` returns the correct index of the maximum
 * element.
 * @return 1 on success, 0 on failure.
 */
int test_matrix_argmax() {
  Matrix* m = create_matrix(1, 5);
  m->matrix_data[0] = 0.1;
  m->matrix_data[1] = 0.9;
  m->matrix_data[2] = 0.2;
  m->matrix_data[3] = 0.8;
  m->matrix_data[4] = 0.5;

  size_t argmax_idx = matrix_argmax(m);
  TEST_ASSERT(argmax_idx == 1, "matrix_argmax failed");

  free_matrix(m);
  return 1;
}

/**
 * @brief Tests the sigmoid activation function.
 * Verifies that the sigmoid function produces correct outputs for given inputs.
 * @return 1 on success, 0 on failure.
 */
int test_sigmoid() {
  Matrix* m = create_matrix(1, 2);
  Matrix* expected = create_matrix(1, 2);
  m->matrix_data[0] = 0;
  m->matrix_data[1] = 1;
  expected->matrix_data[0] = 0.5;
  expected->matrix_data[1] = 1.0 / (1.0 + exp(-1.0));  // approx 0.731

  Matrix* result = sigmoid(m);
  TEST_ASSERT(compare_matrices(result, expected, 1e-9), "sigmoid failed");

  free_matrix(m);
  free_matrix(expected);
  free_matrix(result);
  return 1;
}

/**
 * @brief Tests the basic put and get functionality of the cache.
 * Verifies that matrices can be stored, retrieved, and updated correctly.
 * @return 1 on success, 0 on failure.
 */
int test_cache_functionality() {
  Cache* cache = create_cache();
  TEST_ASSERT(cache != NULL, "create_cache failed");

  Matrix* m1 = create_matrix(1, 1);
  m1->matrix_data[0] = 10.0;
  cache_put(cache, "test_key", m1);  // cache_put takes ownership of m1

  Matrix* retrieved = cache_get(cache, "test_key");
  TEST_ASSERT(retrieved != NULL, "cache_get failed to retrieve matrix");
  TEST_ASSERT(retrieved->matrix_data[0] == 10.0,
              "cached matrix data incorrect");

  // Test updating a cached value
  Matrix* m2 = create_matrix(1, 1);
  m2->matrix_data[0] = 20.0;
  cache_put(cache, "test_key",
            m2);  // cache_put takes ownership of m2, frees old m1

  Matrix* updated_retrieved = cache_get(cache, "test_key");
  TEST_ASSERT(updated_retrieved != NULL,
              "cache_get failed to retrieve updated matrix");
  TEST_ASSERT(updated_retrieved->matrix_data[0] == 20.0,
              "updated cached matrix data incorrect");

  free_matrix(retrieved);
  free_matrix(updated_retrieved);
  free_cache(cache);
  return 1;
}

/**
 * @brief Main function for the test suite.
 * Runs all individual test functions and prints a summary of the results.
 * @return EXIT_SUCCESS if all tests pass, EXIT_FAILURE otherwise.
 */
int main() {
  printf("Starting all tests...\n");

  RUN_TEST(test_create_free_matrix);
  RUN_TEST(test_add_matrix);
  RUN_TEST(test_dot_matrix);
  RUN_TEST(test_transpose_matrix);
  RUN_TEST(test_matrix_argmax);
  RUN_TEST(test_sigmoid);
  RUN_TEST(test_cache_functionality);

  printf("\n--- Test Summary ---\n");
  printf("Total tests run: %d\n", tests_run);
  printf("Tests passed: %d\n", tests_passed);
  printf("Tests failed: %d\n", tests_failed);

  if (tests_failed > 0) {
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
}
