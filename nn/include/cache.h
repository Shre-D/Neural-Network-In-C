#pragma once

#include "linalg.h"

// Forward declaration of the internal cache struct to hide implementation
// details.
typedef struct Cache Cache;

Cache* init_cache();

// A deep copy of the matrix is stored to prevent in-place modification issues.
void put_matrix(Cache* cache, const char* key, const Matrix* m);

// Returns a deep copy of the matrix, or NULL if not found.
Matrix* get_matrix(Cache* cache, const char* key);

void clear_cache(Cache* cache);
