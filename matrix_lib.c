#include <stdio.h>
#include <immintrin.h>

struct matrix { 
    unsigned long int height;
    unsigned long int width;
    float *rows;
};

void test_scalar_matrix_mult();
int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
void printMatrix(struct matrix *matrix);

int main(void) {
    test_scalar_matrix_mult();
    return 0;
}

void test_scalar_matrix_mult() {
    struct matrix testMatrix;
    float *rows;

    rows = malloc(8*sizeof(float));
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
    int result = scalar_matrix_mult(2.0,&testMatrix);

    printMatrix(&testMatrix);
    free(rows);
}

int scalar_matrix_mult(float scalar_value, struct matrix *matrix) {
    unsigned long int height = matrix->height;
    unsigned long int width = matrix->width;
    float *currentRow = matrix->rows;
    __m256 currentVec; 
    __m256 scalarVec;
    __m256 resultVec;

    if(height == 0 || width == 0 || currentRow == NULL) { 
        return 0;
    }

    scalarVec = _mm256_set1_ps(scalar_value);

    for(int i=0; i < (height*width)/8; i++) {
        currentVec = _mm256_loadu_ps(currentRow);
        
        // operation
        resultVec = _mm256_mul_ps(currentVec, scalarVec);
        
        // store the result
        _mm256_store_ps(currentRow, resultVec);
        
        // increment pointer
        currentRow++;
    }

    return 1;
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