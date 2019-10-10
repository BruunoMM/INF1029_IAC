#include <stdio.h>
#include <immintrin.h>
#include <stdlib.h>
#include "matrix_lib.h"

struct matrix *readDatFile(FILE *arq, int height, int width);
void writeDatFile(FILE *arq, struct matrix *matrix);
struct matrix* createMatrixC(int height, int width);
void freeMatrix(struct matrix* matrix);


int main(int argc, char *argv[]) {
    FILE *arq, *arq2, *arq3, *arq4;
    int height1, width1, height2, width2, threads;
    float cons;
    char *pathFile1, *pathFile2, *pathResultFile1, *pathResultFile2;

    cons = atof(argv[1]);
    height1 = atoi(argv[2]);
    width1 = atoi(argv[3]);
    height2 = atoi(argv[4]);
    width2 = atoi(argv[5]);
    threads = atoi(argv[6]);
    pathFile1 = argv[7];
    pathFile2 = argv[8];
    pathResultFile1 = argv[9];
    pathResultFile2 = argv[10];

    if(argc != 11) {
        printf("Numero de argumentos invalidos.\n");
        return -1;
    } else if ((height1 + width1 + height2 + width2) % 8 != 0) {
        printf("Dimensoes nao multiplas de 8.\n");
        return -1;
    }
    
    arq = fopen(pathFile1, "rb");
    arq2 = fopen(pathFile2, "rb");
    arq3 = fopen(pathResultFile1, "wb");
    arq4 = fopen(pathResultFile2, "wb");
    
    if (!arq || !arq2) {
        printf("Arquivos de entrada nao existentes.\n");
        return -1;
    }

    if(!arq3 || !arq4) {
        printf("Nao foi possivel criar arquivos de saida.\n");
        return -1;
    }
    struct matrix *matrixA = readDatFile(arq, height1, width1);
    struct matrix *matrixB = readDatFile(arq2, height2, width2);
    struct matrix *matrixC = createMatrixC(height1, width2);

    set_number_threads(threads);
    scalar_matrix_mult(cons, matrixA);
    writeDatFile(arq3, matrixA);
    
    matrix_matrix_mult(matrixA, matrixB, matrixC);
    writeDatFile(arq4, matrixC);

    freeMatrix(matrixA);
    freeMatrix(matrixB);
    freeMatrix(matrixC);

    fclose(arq);    
    fclose(arq2);   
    fclose(arq3);
    fclose(arq4);
    
    return 0;
}

struct matrix* createMatrixC(int height, int width) {
    struct matrix *matrixC = malloc(sizeof(struct matrix));
    matrixC->rows = aligned_alloc(32, height * width * sizeof(float));
    matrixC->height = height;
    matrixC->width = width;

    return matrixC;
}

void freeMatrix(struct matrix* matrix) {
    free(matrix->rows);
    free(matrix);
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