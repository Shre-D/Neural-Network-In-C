#pragma once

#include <CUnit/Basic.h>

#include "linalg.h"

/**
 * @file test_utils.h
 * @brief Header for utility functions used in tests.
 */

#pragma once

#include <CUnit/Basic.h>

#include "linalg.h"

/**
 * @brief Helper function to compare two matrices for CUnit assertions.
 * @param m1 Pointer to the first matrix.
 * @param m2 Pointer to the second matrix.
 * @param epsilon The maximum allowed difference between corresponding elements.
 * @return 1 if the matrices are approximately equal, 0 otherwise.
 */
int compare_matrices(Matrix* m1, Matrix* m2, double epsilon);
