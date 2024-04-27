#include <stdio.h>

#define M 16
#define N 16
#define K 16
#define TILE_WIDTH 4

__global__ void matrixMul(float *A, float *B, float *C, int m, int n, int k) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    float Cvalue = 0;

    for (int t = 0; t < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (Row < m && t * TILE_WIDTH + tx < k) {
            As[ty][tx] = A[Row * k + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if (t * TILE_WIDTH + ty < k && Col < n) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * n + Col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (Row < m && Col < n) {
        C[Row * n + Col] = Cvalue;
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

   

    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result matrix C from device to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
