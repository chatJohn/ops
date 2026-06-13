#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce_in_place(float *input, const int N){
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads(); // ensure all threads have original data or updated data
        if (tid % (2 * stride) == 0 && (idx + stride) < N){
            input[idx] += input[idx + stride];
        }
    }
    if(tid == 0){
        input[blockIdx.x] = input[blockIdx.x * blockDim.x];
    }
}

float reduce_cpu(float *input, const int N){
    float sum = 0.0f;
    for(int i = 0; i < N; i++){
        sum += input[i];
    }
    return sum;
}

 
int main(){
    int n = 1024 * 1024;
    size_t bytes = n * sizeof(float);
    float *h_input = new float[n];
    float *d_input;
    for(int i = 0; i < n; i++){
        h_input[i] = static_cast<float>(i + 1);
    }
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    float cpu_sum = reduce_cpu(h_input, n);
    std::cout << "CPU Sum: " << cpu_sum << std::endl;
    while(gridSize > 1){
        reduce_in_place<<<gridSize, blockSize>>>(d_input, n);
        cudaDeviceSynchronize();

        n = gridSize;
        gridSize = (n + blockSize - 1) / blockSize;
    }// kernel 1
    //kernel 2
    reduce_in_place<<<1, blockSize>>>(d_input, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_input, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "GPU Sum: " << h_input[0] << std::endl;
    cudaFree(d_input);
    delete[] h_input;
    return 0;
}