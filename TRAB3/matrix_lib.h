#define THREADS_PER_BLOCK 256

struct matrix {
    unsigned long int height;
    unsigned long int width;
    float *d_rows;      // GPU
    float *h_rows;      // CPU
};

void set_number_threads(int num_threads);
int scalar_matrix_mult(float scalar_value, struct matrix *matrix);
int matrix_matrix_mult(struct matrix *matrixA, struct matrix * matrixB, struct matrix * matrixC);
int run_tests();
void printMatrix(struct matrix *matrix);
void mult_scalar(float scalar, int n, struct matrix *matrix);
