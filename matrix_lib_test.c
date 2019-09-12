#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include "matrix_lib.h"

struct matrix *readDatFile(FILE *arq, int height, int width);

int main(int argc, char *argv[]) {
    // run_tests();
    FILE *arq = fopen("floats_256_2.0f.dat", "r");
    FILE *arq2 = fopen("floats_256_5.0f.dat", "r");
    int height1, width1, height2, width2;
    float cons;

    cons = atof(argv[1]);
    height1 = atoi(argv[2]);
    width1 = atoi(argv[3]);
    height2 = atoi(argv[4]);
    width2 = atoi(argv[5]);

    struct matrix *matrixA = readDatFile(arq, height1, width1);
    struct matrix *matrixB = readDatFile(arq2, height2, width2);

    
    return 0;
}

struct matrix *readDatFile(FILE *arq, int height, int width){
    struct matrix *matrixEx;
    float *rows;
    
    matrixEx = malloc(sizeof(struct matrix));
    rows = aligned_alloc(32,32*sizeof(float));
    float* tmpRows = rows;

    while (fscanf(arq,"%lf",&tmpRows) == 1){
        tmpRows++;
    }

    matrixEx->rows = rows;
    printf("%f\n", matrixEx->rows[0]);

    return matrixEx;
}