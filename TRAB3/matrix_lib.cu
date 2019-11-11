#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

#define THREADS_PER_BLOCK 256

void test_scalar_matrix_mult();
void printMatrix(struct matrix *matrix);
void fill_matrix_with_zero(struct matrix *matrix);
void test_matrix_matrix_mult();
void *calculate_scalar_matrix(void *thread_id);
void *calculate_matrix_matrix(void *thread_id);

// scalarMatrix
__m256 currentVec; 
__m256 scalarVec;
__m256 resultVec;
struct matrix *global_matrix;
float scalarValue;

// matrixMatrix
struct matrix *gmatrixA;
struct matrix *gmatrixB;
struct matrix *gmatrixC;

int run_tests(void) {
    printf("\nRunning scalar multiplication test\n");
    test_scalar_matrix_mult();
    printf("\nRunning matrix multiplication test\n");
    test_matrix_matrix_mult();
    printf("\n\n");
    return 0;
}

void test_matrix_matrix_mult() {
    struct matrix testMatrixA, testMatrixB, testMatrixC;
    float *rowsA, *rowsB, *rowsC;

    rowsA = aligned_alloc(32, 32*sizeof(float));
    for(int i = 0; i < 8; i++){
        rowsA[i] = 9.0;
    }

    testMatrixA.height = 2;
    testMatrixA.width = 3;
    testMatrixA.rows = rowsA;

    rowsB = aligned_alloc(32, 32*sizeof(float));
    for(int i = 0; i < 24; i++){
        rowsB[i] = 2.0;
    }

    testMatrixB.height = 3;
    testMatrixB.width = 8;
    testMatrixB.rows = rowsB;

    rowsC = aligned_alloc(32, 32*sizeof(float));
    testMatrixC.height = 2;
    testMatrixC.width = 8;
    testMatrixC.rows = rowsC;

    printMatrix(&testMatrixA);
    printf("---\n");
    printMatrix(&testMatrixB);

    matrix_matrix_mult(&testMatrixA, &testMatrixB, &testMatrixC);
    printf("---\n");
    printMatrix(&testMatrixC);
}

void test_scalar_matrix_mult() {
    struct matrix testMatrix;
    float *rows;

    rows = aligned_alloc(32,32*sizeof(float));
    rows[0] = 5.0;
    rows[1] = 2.0;
    rows[2] = 3.0;
    rows[3] = 1.0;
    rows[4] = 1.0;
    rows[5] = 1.0;
    rows[6] = 1.0;
    rows[7] = 1.0;

    testMatrix.height = 2;
    testMatrix.width = 4;
    testMatrix.rows = rows;
    
    printMatrix(&testMatrix);
    printf("----\n");
    scalar_matrix_mult(2.0,&testMatrix);

    printMatrix(&testMatrix);
    free(rows);
}
__global__
void mult_scalar(float scalar, int n, struct matrix *matrix) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(i = index; i < n; i += stride){
        matrix->d_rows[i] *= scalar;
    }
}
int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    unsigned long int height = matrix->height;
    unsigned long int width = matrix->width;

    global_matrix = matrix;
    scalarValue = scalar_value;

    if(height == 0 || width == 0 || matrix->rows == NULL) {
        printf("Dimensao nao pode ser igual a zero.\n"); 
        return 0;
    }
    return 1;
}


int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long int heightA = matrixA->height;
    unsigned long int widthA = matrixA->width, widthB = matrixB->width;
    pthread_t *threads;
    pthread_attr_t attribute;

    gmatrixA = matrixA;
    gmatrixB = matrixB;
    gmatrixC = matrixC;

    if(heightA == 0 || widthA == 0 || widthB == 0) {
        printf("Dimensao nao pode ser igual a zero.\n"); 
        return 0;
    }
    return 1;
}

void printMatrix(struct matrix *matrix) {
    int width = matrix->width;
    int height = matrix->height;

    for(int i=1; i <= width * height; i++) {
        printf("%.2f \t", matrix->d_rows[i-1]);
        if(!(i % width)) {
            printf("\n");
        }
    }
}
