#pragma once

#include "linalg.h"

/**
 * @file cache.h
 * @brief Lightweight key-value cache for matrices used across passes.
 *
 * Stores deep copies of matrices keyed by strings. Retrieval returns a deep
 * copy as well, to guard against in-place modification bugs between forward and
 * backward passes.
 */

/** @brief Opaque cache type; implementation hidden. */
typedef struct Cache Cache;

/** @brief Create an empty cache. */
Cache* init_cache();

/** @brief Store a deep copy of matrix `m` under key `key`. */
void put_matrix(Cache* cache, const char* key, const Matrix* m);

/** @brief Retrieve a deep copy of a matrix by key, or NULL if not found. */
Matrix* get_matrix(Cache* cache, const char* key);

/** @brief Free all entries and the cache itself. */
void clear_cache(Cache* cache);
