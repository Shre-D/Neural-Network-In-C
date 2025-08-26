#pragma once

// _matrix is used to avoid confusion if the struct is self referencing
// 1D array used, this is way more performant than a 2D array is
typedef struct _Matrix {
  double* matrix_data;
  int rows;
  int cols;
} Matrix;

//==============================
// Thought I'd comment my stuff nicely for once.
// I'm going to define functions on matrices here,
// and group them according to function
//==============================

//=====================
// Functions for Matrix IO
//=====================

Matrix* read_matrix(const char* fileName);
Matrix* create_matrix(int rows, int cols);
Matrix* copy_matrix(const Matrix* m);
Matrix* flatten_matrix(Matrix* m, int axis);
void fill_matrix(Matrix* m, double n);
void randomize_matrix(Matrix* m, double n);
void free_matrix(Matrix* m);
void write_matrix(Matrix* m,const char* filename);
void print_matrix(Matrix* m);
int matrix_argmax(Matrix* m);

//============================
// Functions for Matrix Operations
//============================
Matrix* identity_matrix(int n);
Matrix* add_matrix(Matrix* a, Matrix* b);
Matrix* subtract_matrix(Matrix* a, Matrix* b);
Matrix* multiply_matrix(Matrix* a, Matrix* b);
Matrix* apply_onto_matrix(double* (*func)(double),
                          Matrix* m);  // Apply our own stuff onto the matrix
                                       // items, useful for activation
Matrix* add_scalar_to_matrix(Matrix* m, double n);
Matrix* dot_matrix(Matrix* a, Matrix* b);
Matrix* transpose_matrix(Matrix* m);
Matrix* scale_matrix(double n,
                     Matrix* m);  // matrix can be scaled to non integer values
