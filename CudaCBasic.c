#include <stdio.h>

#define M 16
#define N 16
#define K 16

__global__ void matrixMul(float *A, float *B, float *C, int m, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < m && j < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; ++l) {
            sum += A[i * k + l] * B[l * n + j];
        }
        C[i * n + j] = sum;
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

    
I
    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
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
