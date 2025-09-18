/**
 * @file cunit_runner.c
 * @brief Main test runner for CUnit tests.
 * This file initializes the CUnit test registry, adds test suites, and runs all
 * tests.
 */

#include <CUnit/Basic.h>
#include <stdio.h>
#include <stdlib.h>

// Declare test suites from other test files
extern CU_TestInfo core_tests[];
extern CU_TestInfo nn_tests[];

/**
 * @brief Adds a suite of tests to the CUnit registry.
 * @param suite_name The name of the test suite.
 * @param tests An array of CU_TestInfo structures defining the tests in the
 * suite.
 * @return 0 on success, or a CUnit error code on failure.
 */
int add_suite(const char* suite_name, CU_TestInfo tests[]) {
  CU_pSuite pSuite = NULL;
  pSuite = CU_add_suite(suite_name, NULL, NULL);
  if (NULL == pSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }
  for (int i = 0; tests[i].pName != NULL; i++) {
    if (NULL == CU_add_test(pSuite, tests[i].pName, tests[i].pTestFunc)) {
      CU_cleanup_registry();
      return CU_get_error();
    }
  }
  return 0;
}

/**
 * @brief Main function for running CUnit tests.
 * Initializes the registry, adds all defined test suites, runs tests, and
 * cleans up.
 * @return The CUnit error code after test execution.
 */
int main() {
  // Initialize the CUnit test registry
  if (CUE_SUCCESS != CU_initialize_registry()) {
    return CU_get_error();
  }

  // Add suites
  if (add_suite("Core Tests", core_tests) != 0) return CU_get_error();
  if (add_suite("Neural Network Tests", nn_tests) != 0) return CU_get_error();

  // Run all tests using the Basic interface
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();

  // Clean up registry and return results
  CU_cleanup_registry();
  return CU_get_error();
}
