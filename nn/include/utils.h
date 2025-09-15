#pragma once

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @file utils.h
 * @brief Logging and assertion utilities used across modules.
 */

// Set up log levels
typedef enum {
  LOG_LEVEL_DEBUG,
  LOG_LEVEL_INFO,
  LOG_LEVEL_WARN,
  LOG_LEVEL_ERROR
} LogLevel;

// Set the minimum log level to display. Change this to filter output.
#define MIN_LOG_LEVEL LOG_LEVEL_INFO

// Assert macro, it'll be useful for checking matrix properties
#define ASSERT(condition, message)                                         \
  do {                                                                     \
    if (!(condition)) {                                                    \
      fprintf(stderr, "Assertion failed in %s at line %d: %s\n", __FILE__, \
              __LINE__, message);                                          \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// For safe malloc allocations
#define CHECK_MALLOC(ptr, message) ASSERT((ptr) != NULL, message);

// Macro to handle memory allocation checks.
#define CHECK_ALLOC(ptr) \
  if (ptr == NULL) {     \
    return NULL;         \
  }

// Function prototypes for logging
void log_message(LogLevel level, const char* format, ...);

// Convenience macros for easier logging
#define LOG_DEBUG(format, ...) \
  log_message(LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_message(LOG_LEVEL_INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_message(LOG_LEVEL_WARN, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) \
  log_message(LOG_LEVEL_ERROR, format, ##__VA_ARGS__)