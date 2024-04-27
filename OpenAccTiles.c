#include <stdio.h>

#define M 16
#define N 16
#define K 16
#define TILE_WIDTH 4

void matrixMul(float *A, float *B, float *C, int m, int n, int k) {
    #pragma acc parallel loop present(A[0:m*k], B[0:k*n], C[0:m*n])
    for (int i = 0; i < m; i++) {
        #pragma acc loop present(A[0:m*k], B[0:k*n], C[0:m*n])
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    float *A, *B, *C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate memory on host
    A = (float*)malloc(size_A);
    B = (float*)malloc(size_B);
    C = (float*)malloc(size_C);

  
    // Perform matrix multiplication using OpenACC
    matrixMul(A, B, C, M, N, K);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
