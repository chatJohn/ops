#include <stdio.h>
#include <cstdlib>
#define N 4096
#define M 4096

__global__ void matrix_add(float *A, float *B, float *C, const int NN, const int MM){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < NN && col < MM){
        int idx = row * MM + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    int size = N * M * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for(int i = 0; i < N * M; i++){
        h_A[i] = 10 * static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        h_B[i] = 10 * static_cast <float> (rand()) / static_cast<float> (RAND_MAX);
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    dim3 blockDim(32, 32);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    cudaFuncSetCacheConfig(matrix_add, cudaFuncCachePreferL1);
    matrix_add<<<gridDim, blockDim>>> (d_A, d_B, d_C, N, M);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("A[i] + B[i] = C[i]\n");
    printf("%f + %f = %f\n", h_A[0], h_B[0], h_C[0]);
    printf("%f + %f = %f\n", h_A[1], h_B[1], h_C[1]);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0; 
}   