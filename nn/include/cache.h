#pragma once

#include "linalg.h"

// Forward declaration of the internal cache struct to hide implementation details.
typedef struct Cache Cache;

// Initializes and allocates memory for a new cache.
// Must be called once before use. Returns a pointer to the new cache instance.
Cache* init_cache();

// Stores a matrix in the cache with a given key.
// A deep copy of the matrix is stored to prevent in-place modification issues.
void put_matrix(Cache* cache, const char* key, const Matrix* m);

// Retrieves a matrix from the cache by its key.
// Returns a deep copy of the matrix, or NULL if not found.
Matrix* get_matrix(Cache* cache, const char* key);

// Clears all matrices from the cache and frees their memory.
void clear_cache(Cache* cache);
