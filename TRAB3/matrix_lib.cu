#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix_lib.h"

void printMatrix(struct matrix *matrix);

__global__
void mult_scalar(float scalar, int n, float* d_rows) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < n; i += stride){
        d_rows[i] *= scalar;
    }
}

/*__global__
void mult_matrix(float *d_rowsA, float *d_rowsB, float *d_rowsC, int heightA, int widthA, int widthB){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    
    float soma = 0.0;
    if(row < heightA && column < widthB){
       for(int k = 0; k < widthA; k++){
          soma += d_rowsA[row*widthA+k] * d_rowsB[k*widthB + column];
       }
       d_rowsC[row*widthB+column] = soma;
    }
}	*/
__global__
void mult_matrix(float* a_rows, float* b_rows, float *c_rows, int heightA, int widthA, int widthB) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int nA = widthA * heightA; // Numero de elementos de A
    int currentLineC = index / widthB;     
    int indColunaB = (index - currentLineC * widthB) % widthB; // Indice da coluna de B sendo utilizada
    int currentLineA = (index % nA) / widthA;
    int indComecoA = currentLineA*widthA; // Indice de começo de uma determinada linha na matriz A 
    int posC = index % widthB*heightA; // Posicao na matriz C
 
    for(int i = 0; i < widthA; i++)	 
    	c_rows[posC] += a_rows[indComecoA + i] * b_rows[indColunaB + i*widthB];
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

    int blockSize = THREADS_PER_BLOCK;
    int numBlocks = (height * width + blockSize - 1) / blockSize;

    mult_scalar<<<numBlocks, blockSize>>>(scalar_value, height*width, matrix->d_rows);
    safeCudaMemCpy(matrix->h_rows, matrix->d_rows, height*width, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    return 1;
}


int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long int heightA = matrixA->height, heightB = matrixB->height;
    unsigned long int widthA = matrixA->width, widthB = matrixB->width;

    if(heightA == 0 || widthA == 0 || widthB == 0 || widthA != heightB) {
        printf("Dimensao nao pode ser igual a zero.\n"); 
        return 0;
    }
    
    int blockSize = THREADS_PER_BLOCK;
    unsigned int grid_rows = (heightA + blockSize - 1) / blockSize;
    unsigned int grid_cols = (widthB + blockSize -1) / blockSize;
    dim3 dimGrid(grid_rows, grid_cols);
    dim3 dimBlocks(blockSize, blockSize);
    
    mult_matrix<<<dimGrid, dimBlocks>>>(matrixA->d_rows, matrixB->d_rows, matrixC->d_rows, heightA, widthA, widthB);
    safeCudaMemCpy(matrixC->h_rows, matrixC->d_rows, heightA * widthB, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
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
