#include "utils.h"

#include <stdarg.h>
#include <string.h>

const char* get_log_level_string(LogLevel level) {
    switch (level) {
        case LOG_LEVEL_DEBUG: return "DEBUG";
        case LOG_LEVEL_INFO:  return "INFO";
        case LOG_LEVEL_WARN:  return "WARN";
        case LOG_LEVEL_ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void log_message(LogLevel level, const char* format, ...) {

    if (level < MIN_LOG_LEVEL) {
        return;
    }

    // include time
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char time_str[20];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", t);

    FILE* stream = (level >= LOG_LEVEL_WARN) ? stderr : stdout;

    va_list args;
    va_start(args, format); // sets pointer to the beginning of our args
    
    fprintf(stream, "[%s] [%s] ", time_str, get_log_level_string(level));
    vfprintf(stream, format, args);
    fprintf(stream, "\n");
    
    va_end(args); // cleans up va args memory
}