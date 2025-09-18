/**
 * @file utils.c
 * @brief Implementation of logging and assertion utilities.
 */

#include "utils.h"

#include <stdarg.h>
#include <string.h>

/**
 * @brief Returns the string representation of a given LogLevel.
 * @param level The LogLevel enum value.
 * @return A string literal representing the log level.
 */
const char* get_log_level_string(LogLevel level) {
  switch (level) {
    case LOG_LEVEL_DEBUG:
      return "DEBUG";
    case LOG_LEVEL_INFO:
      return "INFO";
    case LOG_LEVEL_WARN:
      return "WARN";
    case LOG_LEVEL_ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief Logs a formatted message to stdout or stderr based on the log level.
 * Messages with level WARN or ERROR are directed to stderr.
 * @param level The LogLevel of the message.
 * @param format The format string for the message.
 * @param ... Variable arguments to be formatted according to 'format'.
 */
void log_message(LogLevel level, const char* format, ...) {
  if (level < MIN_LOG_LEVEL) {
    return;
  }

  // include time
  time_t now = time(NULL);
  const struct tm* t = localtime(&now);
  char time_str[20];
  strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", t);

  FILE* stream = (level >= LOG_LEVEL_WARN) ? stderr : stdout;

  va_list args;
  va_start(args, format);  // sets pointer to the beginning of our args

  fprintf(stream, "[%s] [%s] ", time_str, get_log_level_string(level));
  vfprintf(stream, format, args);
  fprintf(stream, "\n");

  va_end(args);  // cleans up va args memory
}
