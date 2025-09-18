#include <CUnit/Basic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cache.h"
#include "linalg.h"
#include "test_utils.h"
#include "utils.h"

/**
 * @file core_cunit.c
 * @brief CUnit tests for core linear algebra and cache functionalities.
 */

#include <CUnit/Basic.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cache.h"
#include "linalg.h"
#include "test_utils.h"
#include "utils.h"

/**
 * @brief Tests the creation and freeing of a matrix.
 * Verifies that `create_matrix` allocates memory correctly and `free_matrix`
 * deallocates it without issues.
 */
void test_create_free_matrix() {
  Matrix* m = create_matrix(2, 3);
  CU_ASSERT_PTR_NOT_NULL(m);
  CU_ASSERT_EQUAL(m->rows, 2);
  CU_ASSERT_EQUAL(m->cols, 3);
  free_matrix(m);
}

/**
 * @brief Tests the element-wise addition of two matrices.
 * Verifies that `add_matrix` correctly sums corresponding elements of two
 * matrices.
 */
void test_add_matrix() {
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
  CU_ASSERT_TRUE(compare_matrices(result, expected, 1e-9));

  free_matrix(a);
  free_matrix(b);
  free_matrix(expected);
  free_matrix(result);
}

/**
 * @brief Tests the functionality of the cache (put and get operations).
 * Verifies that matrices can be stored and retrieved correctly from the cache,
 * and that updates work as expected.
 */
void test_cache_functionality() {
  Cache* cache = create_cache();
  CU_ASSERT_PTR_NOT_NULL(cache);

  Matrix* m1 = create_matrix(1, 1);
  m1->matrix_data[0] = 10.0;
  cache_put(cache, "test_key", m1);  // cache_put takes ownership of m1

  Matrix* retrieved = cache_get(cache, "test_key");
  CU_ASSERT_PTR_NOT_NULL(retrieved);
  CU_ASSERT_DOUBLE_EQUAL(retrieved->matrix_data[0], 10.0, 1e-9);

  // Test updating a cached value
  Matrix* m2 = create_matrix(1, 1);
  m2->matrix_data[0] = 20.0;
  cache_put(cache, "test_key",
            m2);  // cache_put takes ownership of m2, frees old m1

  Matrix* updated_retrieved = cache_get(cache, "test_key");
  CU_ASSERT_PTR_NOT_NULL(updated_retrieved);
  CU_ASSERT_DOUBLE_EQUAL(updated_retrieved->matrix_data[0], 20.0, 1e-9);

  free_matrix(retrieved);
  free_matrix(updated_retrieved);
  free_cache(cache);
}

/**
 * @brief Array of CU_TestInfo structures for core tests.
 */
CU_TestInfo core_tests[] = {
    {"test_create_free_matrix", test_create_free_matrix},
    {"test_add_matrix", test_add_matrix},
    {"test_cache_functionality", test_cache_functionality},
    CU_TEST_INFO_NULL};
