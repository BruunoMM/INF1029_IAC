#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include "matrix_lib.h"

struct matrix *readDatFile(FILE *arq, int height, int width);
void writeDatFile(FILE *arq, struct matrix *matrix);

int main(int argc, char *argv[]) {
    // run_tests();
    FILE *arq, *arq2;
    int height1, width1, height2, width2;
    float cons;
    char *pathFile1, *pathFile2, *pathResultFile1, *pathResultFile2;

    cons = atof(argv[1]);
    height1 = atoi(argv[2]);
    width1 = atoi(argv[3]);
    height2 = atoi(argv[4]);
    width2 = atoi(argv[5]);
    pathFile1 = argv[6];
    pathFile2 = argv[7];
    pathResultFile1 = argv[8];
    pathResultFile2 = argv[9];

    if(argc != 10) {
        printf("Numero de argumentos invalidos.\n");
        return -1;
    } else if ((height1 + width1 + height2 + width2) % 8 != 0) {
        printf("Dimensoes nao multiplas de 8.\n");
        return -1;
    }
    
    arq = fopen(pathFile1, "rb");
    arq2 = fopen(pathFile2, "rb");
    FILE* arq3 = fopen("result1.dat", "wb");
    
    if (!arq || !arq2 || !arq3) {
        printf("Arquivos de entrada nao existentes.\n");
        return -1;
    }

    struct matrix *matrixA = readDatFile(arq, height1, width1);
    struct matrix *matrixB = readDatFile(arq2, height2, width2);
    struct matrix *matrixC = malloc(sizeof(struct matrix));
    matrixC->rows = aligned_alloc(32, height1 * width2 * sizeof(float));
    matrixC->height = height1;
    matrixC->width = width2;

    scalar_matrix_mult(cons, matrixA);

    printMatrix(matrixA);
    
    matrix_matrix_mult(matrixA, matrixB, matrixC);

    printMatrix(matrixC);
    return 0;
}

struct matrix *readDatFile(FILE *arq, int height, int width){
    struct matrix *matrixEx;
    float *rows;
    int count = 0;
    int totalSize = height * width;
    
    matrixEx = malloc(sizeof(struct matrix));
    rows = aligned_alloc(32,totalSize*sizeof(float));

    matrixEx->height = height;
    matrixEx->width = width;
    while (!feof(arq) && (count < totalSize)){
        fread(&rows[count], sizeof(float), 1, arq);
        count++;
    }

    matrixEx->rows = rows;

    return matrixEx;
}

void writeDatFile(FILE *arq, struct matrix *matrix) {
    int height = matrix->height;
    int width = matrix->width;
    int totalSize = height * width;

    for(int i=0 ; i < totalSize ; i++) {
        float numberToWrite = matrix->rows[i];
        int result = fwrite(&numberToWrite, sizeof(float), 1, arq);
        if(result != 1) {
            printf("Erro de escrita!!! \n");
        }
    }
}