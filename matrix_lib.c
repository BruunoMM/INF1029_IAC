#include <stdio.h>
#include <immintrin.h>

int main(void) {

    // write scalar_matrix_mult tests
    return 0;
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

    for(int i=0; i < (height*width)/8; i++) {
        currentVec = _mm256_loadu_ps(currentRow);
        scalarVec = _mm256_set1_ps(scalar_value);

        // operation
        resultVec = _mm256_mul_ps(currentVec, scalarVec);
        
        // store the result
        _mm256_store_pd(currentRow, resultVec);
        
        // increment pointer
        currentRow++;
    }

    return 1;
}

struct matrix { 
    unsigned long int height;
    unsigned long int width;
    float *rows;
}