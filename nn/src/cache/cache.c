/**
 * @file cache.c
 * @brief Internal hashmap-based cache for matrices.
 */
#include "cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"

// A linked list is used to handle collisions at each bucket.
/**
 * @brief Represents an entry in the cache, storing a key-matrix pair.
 * Uses a linked list for collision resolution in the hash map.
 */
typedef struct CacheEntry {
  char* key; /**< The key associated with the matrix. */
  Matrix* m; /**< The matrix stored in this entry. */
  struct CacheEntry*
      next; /**< Pointer to the next entry in case of collision. */
} CacheEntry;

#define HASH_MAP_SIZE 1024

struct Cache {
  CacheEntry* entries[HASH_MAP_SIZE]; /**< Array of pointers to CacheEntry,
                                         forming the hash table buckets. */
};

/**
 * @brief Computes a hash value for a given string key.
 * @param key The string key to hash.
 * @return An unsigned integer hash value, modulo HASH_MAP_SIZE.
 */
static unsigned int hash(const char* key) {
  // This hash function can be manipulated. You could input a key
  // that could overflow a 32 bit uint, so I've made it a 64 bit
  // number, to make it more robust to attacks. Of course, it's
  // not the most secure, but I'll look into a better hash function
  // later.
  unsigned long long hash = 0;
  size_t i = 0;
  while (key[i] != '\0') {
    hash = hash * 31 + key[i];
    i++;
  }
  return (unsigned int)(hash % HASH_MAP_SIZE);
}

//------------------------------
// Cache Functions
//------------------------------

/**
 * @brief Creates and initializes a new, empty cache.
 * @return A pointer to the newly created Cache, or NULL if memory allocation
 * fails.
 */
Cache* create_cache() {
  Cache* cache = (Cache*)malloc(sizeof(Cache));
  if (cache == NULL) {
    return NULL;
  }

  for (size_t i = 0; i < HASH_MAP_SIZE; i++) {
    cache->entries[i] = NULL;
  }
  return cache;
}

/**
 * @brief Stores a matrix in the cache under a specified key.
 * If the key already exists, the old matrix is freed and replaced with the new
 * one. The cache takes ownership of the matrix `m`.
 * @param cache A pointer to the Cache structure.
 * @param key The string key for the matrix.
 * @param m A pointer to the Matrix to be stored.
 */
void cache_put(Cache* cache, const char* key, Matrix* m) {
  if (cache == NULL || key == NULL || m == NULL) {
    return;
  }

  unsigned int index = hash(key);

  CacheEntry* current = cache->entries[index];
  while (current != NULL) {
    if (strcmp(current->key, key) == 0) {
      // Key found, free existing matrix and update with new one.
      free_matrix(current->m);
      current->m = m;
      return;
    }
    current = current->next;
  }

  CacheEntry* new_entry = (CacheEntry*)malloc(sizeof(CacheEntry));
  if (new_entry == NULL) {
    return;
  }
  new_entry->key = strdup(key);
  new_entry->m = m;
  new_entry->next = cache->entries[index];
  cache->entries[index] = new_entry;
}

/**
 * @brief Retrieves a deep copy of a matrix from the cache using its key.
 * @param cache A pointer to the Cache structure.
 * @param key The string key of the matrix to retrieve.
 * @return A deep copy of the stored Matrix, or NULL if the key is not found or
 * allocation fails. The caller is responsible for freeing the returned matrix.
 */
Matrix* cache_get(Cache* cache, const char* key) {
  if (cache == NULL || key == NULL) {
    return NULL;
  }
  unsigned int index = hash(key);
  CacheEntry* current = cache->entries[index];
  while (current != NULL) {
    if (strcmp(current->key, key) == 0) {
      return copy_matrix(current->m);
    }
    current = current->next;
  }
  return NULL;
}

/**
 * @brief Clears all entries from the cache, freeing associated memory for keys
 * and matrices. The cache structure itself is not freed.
 * @param cache A pointer to the Cache structure to clear.
 */
void clear_cache(Cache* cache) {
  if (cache == NULL) {
    return;
  }
  for (size_t i = 0; i < HASH_MAP_SIZE; i++) {
    CacheEntry* current = cache->entries[i];
    while (current != NULL) {
      CacheEntry* to_free = current;
      current = current->next;
      free(to_free->key);
      free_matrix(to_free->m);
      free(to_free);
    }
    cache->entries[i] = NULL;
  }
}

/**
 * @brief Frees all entries in the cache and the cache structure itself.
 * @param cache A pointer to the Cache structure to free.
 */
void free_cache(Cache* cache) {
  if (cache == NULL) {
    return;
  }
  clear_cache(cache);
  free(cache);
}
