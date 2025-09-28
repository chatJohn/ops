

__global__ void reduce0(float *d_A, const int N){
    extern __shared__ float data[];
    // 从global memory读取数据到shared memory
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = index < N ? d_A[index] : 0.f;
    __syncthreads();

    for(int s = 1; s < blockDim.x; s <<= 1){
        if((tid % (s * 2)) == 0){
            data[tid] += data[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_A[blockIdx.x] = data[0];
    }
}

__global__ void reduce1(float *d_A, const int N){
    extern __shared__ float data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = idx < N ? d_A[idx] : 0.f;
    __syncthreads();

    for(int s = 1; s < blockDim.x; s <<= 1){
        int index = tid * s * 2;
        if(index < blockDim.x){
            data[index] += data[index + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        d_A[blockIdx.x] = data[0];
    }
}

__global__ void reduce2(float *d_A, const int N){
    extern __shared__ float data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = idx < N ? d_A[idx] : 0.f;
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(tid < s){
            data[tid] += data[tid + s];
        } 
        __syncthreads();
    }
    if(tid == 0){
        d_A[blockIdx.x] = data[0];
    }
}

// 在读取的时候就进行规约
__global__ void reduce3(float *d_A, const int N){
    extern __shared__ float data[];
    int tid = threadIdx.x;
    int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    float sum = idx < N ? d_A[idx] : 0.f;
    if(idx + blockDim.x < N){
        data[tid] = sum + d_A[idx + blockDim.x];
    }
    __syncthreads();

    for(int s = blockDim.x >> 1; s > 0; s >>= 1){
        if(tid < s){
            data[tid] += data[tid + s];
        } 
        __syncthreads();
    }
    if(tid == 0){
        d_A[blockIdx.x] = data[0];
    }
}


