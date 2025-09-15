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
typedef struct CacheEntry {
  char* key;
  Matrix* m;
  struct CacheEntry* next;
} CacheEntry;

#define HASH_MAP_SIZE 1024

struct Cache {
  CacheEntry* entries[HASH_MAP_SIZE];
};

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

void cache_put(Cache* cache, const char* key, const Matrix* m) {
  if (cache == NULL || key == NULL || m == NULL) {
    return;
  }

  unsigned int index = hash(key);

  CacheEntry* current = cache->entries[index];
  while (current != NULL) {
    if (strcmp(current->key, key) == 0) {
      // Key found, free existing matrix and update with new one.
      free_matrix(current->m);
      current->m = copy_matrix(m);  // Deep copy
      return;
    }
    current = current->next;
  }

  CacheEntry* new_entry = (CacheEntry*)malloc(sizeof(CacheEntry));
  if (new_entry == NULL) {
    return;
  }
  new_entry->key = strdup(key);
  new_entry->m = copy_matrix(m);  // Deep copy to prevent side effects
  new_entry->next = cache->entries[index];
  cache->entries[index] = new_entry;
}

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

void free_cache(Cache* cache) {
  if (cache == NULL) {
    return;
  }
  clear_cache(cache);
  free(cache);
}
