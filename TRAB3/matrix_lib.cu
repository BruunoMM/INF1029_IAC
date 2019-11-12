#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

#define THREADS_PER_BLOCK 256

void printMatrix(struct matrix *matrix);

__global__
void mult_scalar(float scalar, int n, float* d_rows) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride){
        d_rows[i] *= scalar;
    }
}

__global__
void mult_matrix(int heightA, int widthA, int heightB, int widthB, float* a_rows, float* b_rows) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    
}

void safeCudaMemCpy(float *d_x, float *h_x, int size, enum cudaMemcpyKind kind) {
    cudaError_t cudaResult;
    cudaResult = cudaMemcpy(d_x, h_x, size*sizeof(float), kind);

    if (cudaResult != cudaSuccess) {
	    printf("cudaMemcpy (h_x -> d_x) returned error %s (code %d), line(%d)\n", cudaGetErrorString(cudaResult), cudaResult, __LINE__);
        exit(1);
    }
}

void safeCudaMalloc(float **ptr, int size) {
    cudaError_t cudaResult; 
    cudaResult = cudaMalloc(ptr, size * sizeof(float));
    
    if (cudaResult != cudaSuccess) {
	    printf("cudaMalloc d_x returned error %s (code %d)\n", cudaGetErrorString(cudaResult), cudaResult);
        exit(1);
    }
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    unsigned long int height = matrix->height;
    unsigned long int width = matrix->width;

    if(height == 0 || width == 0 || matrix->h_rows == NULL) {
        printf("Dimensao nao pode ser igual a zero.\n"); 
        return 0;
    }
    mult_scalar<<<numBlocks, blockSize>>>(cons, height*width, matrix->d_rows);
    safeCudaMemCpy(matrix->h_rows, matrix->d_rows, height*width, cudaMemcpyDeviceToHost);

    return 1;
}


int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long int heightA = matrixA->height;
    unsigned long int widthA = matrixA->width, widthB = matrixB->width;

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
        printf("%.2f \t", matrix->h_rows[i-1]);
        if(!(i % width)) {
            printf("\n");
        }
    }
}
