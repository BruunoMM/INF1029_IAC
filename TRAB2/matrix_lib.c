#include <stdio.h>
#include <immintrin.h>
#include <pthread.h>
#include "matrix_lib.h"

void test_scalar_matrix_mult();
void printMatrix(struct matrix *matrix);
void fill_matrix_with_zero(struct matrix *matrix);
void test_matrix_matrix_mult();
void *calculate_scalar_matrix(void *thread_id);

int threads_num = 1;

__m256 currentVec; 
__m256 scalarVec;
__m256 resultVec;
struct matrix *global_matrix;
float scalarValue;

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

void set_number_threads(int num_threads) {
    threads_num = num_threads;
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    unsigned long int height = matrix->height;
    unsigned long int width = matrix->width;
    pthread_t *threads;
    pthread_attr_t attribute;

    global_matrix = matrix;
    scalarValue = scalar_value;

    if(height == 0 || width == 0 || matrix->rows == NULL) {
        printf("Dimensao nao pode ser igual a zero.\n"); 
        return 0;
    }

    threads = malloc(threads_num * sizeof(pthread_t));
    pthread_attr_init(&attribute);
    pthread_attr_setdetachstate(&attribute, PTHREAD_CREATE_JOINABLE);

    for(long i=0; i < threads_num; i++) {
        pthread_create(&threads[i], &attribute, calculate_scalar_matrix, (void *)i);
    }

    for(long i=0; i < threads_num; i++) {
        pthread_join(threads[i], (void *)&i);
    }

    return 1;
}

void *calculate_scalar_matrix(void *thread_id) {
    unsigned long int height = global_matrix->height;
    unsigned long int width = global_matrix->width;

    int linesToCalculate = (height/threads_num);
    long startingLine = (long)thread_id * linesToCalculate; // not an index
    long startingPoint = startingLine * width;  // index
    
    float *currentRow =  &global_matrix->rows[startingPoint];

    scalarVec = _mm256_set1_ps(scalarValue);

    for(int i=0; i < linesToCalculate; i++) {
        for(int j=0; j < width/8; j++) {
            currentVec = _mm256_load_ps(currentRow);

            printf("THREAD %d - %p\n", (long)thread_id, currentRow);
        
            // operation
            resultVec = _mm256_mul_ps(currentVec, scalarVec);
            
            // store the result
            _mm256_store_ps(currentRow, resultVec);
            
            // increment pointer
            currentRow += 8;
        }
    }
    // pthread_exit(thread_id);
}

int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC) {
    unsigned long int heightA = matrixA->height;
    unsigned long int widthA = matrixA->width, widthB = matrixB->width;
    float *currentPointB = matrixB->rows, *currentPointC = matrixC->rows;

    __m256 matrixAVec;
    __m256 matrixBVec;
    __m256 resultVec;

    if(heightA == 0 || widthA == 0 || widthB == 0 || currentPointB == NULL || currentPointC == NULL) {
        printf("Dimensao nao pode ser igual a zero.\n"); 
        return 0;
    }

    fill_matrix_with_zero(matrixC);
   
    float value;
    for(int i=0; i < (widthA * heightA); i++) {
        value = matrixA->rows[i];
        matrixAVec = _mm256_set1_ps(value); // set value in matrix A to vector
        
        if(i % widthA == 0) {
            currentPointB = matrixB->rows;
        }
        
        int currentLine = i / widthA;
        currentPointC = matrixC->rows + (currentLine * widthB);
        resultVec = _mm256_load_ps(currentPointC);
       
        for(int j=0; j < widthB/8; j++) {
            matrixBVec = _mm256_load_ps(currentPointB);
            resultVec = _mm256_fmadd_ps(matrixAVec, matrixBVec, resultVec);
            
            _mm256_store_ps(currentPointC, resultVec);

            currentPointC += 8;
            currentPointB += 8;

            resultVec = _mm256_load_ps(currentPointC);
        }
    }

    return 1;
}

void fill_matrix_with_zero(struct matrix *matrix) {
    for(int i=0; i < (matrix->height * matrix->width); i++) {
        matrix->rows[i] = 0;
    }
}

void printMatrix(struct matrix *matrix) {
    int width = matrix->width;
    int height = matrix->height;

    for(int i=1; i <= width * height; i++) {
        printf("%.2f \t", matrix->rows[i-1]);
        if(!(i % width)) {
            printf("\n");
        }
    }
}
