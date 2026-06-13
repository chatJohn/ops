template<int BLOCK_SIZE = 128>
__global__ void reduce_v0(float* input, float* output){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = input[idx];
    __syncthreads();
    for(int s = 1; s < BLOCK_SIZE; s *= 2){
        // 出现wrap divergence
        // if(threadIdx.x % (2 * s) == 0){
        //     sdata[threadIdx.x] += sdata[threadIdx.x + s];
        // }
        // 解决wrap divergence
        int index = 2 * s * threadIdx.x;
        if(index < BLOCK_SIZE){{
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        output[blockIdx.x] = sdata[0];
    }
}

// 解决bank冲突
template<int BLOCK_SIZE = 128>
__global__ void reduce_v1(float* input, float* output){
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = input[idx];
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s /= 2){
        if(threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        output[blockIdx.x] = sdata[0];
    }
}

// 解决idle线程
template<int BLOCK_SIZE = 128>
__global__ void reduce_v2(float* input, float* output){
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = input[idx] + input[idx + blockDim.x];
    __syncthreads();
   for(int s = blockDim.x / 2; s > 0; s /= 2){
        if(threadIdx.x < s){
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        output[blockIdx.x] = sdata[0];
    } 
}

// 手动循环展开
template<unsigned int BLOCK_SIZE>
__device__ void wrapReduce(float* sdata, int tid){
    if(BLOCK_SIZE >= 64){
        sdata[tid] += sdata[tid + 32];
    }
    if(BLOCK_SIZE >= 32){
        sdata[tid] += sdata[tid + 16];
    }
    if(BLOCK_SIZE >= 16){
        sdata[tid] += sdata[tid + 8];
    }
    if(BLOCK_SIZE >= 8){
        sdata[tid] += sdata[tid + 4]; 
    }
    if(BLOCK_SIZE >= 4){
        sdata[tid] += sdata[tid + 2]; 
    }
    if(BLOCK_SIZE >= 2){
        sdata[tid] += sdata[tid + 1]; 
    }
}

template<unsigned int BLOCK_SIZE>
__global__ void reduce_v3(float* input, float* output){
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = input[idx] + input[idx + blockDim.x];
    __syncthreads();
    if(BLOCK_SIZE >= 512){
        if(tid < 256){
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE >= 256){
        if(tid < 128){
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if(BLOCK_SIZE >= 128){
        if(tid < 64){
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }
    if(tid < 32){
        wrapReduce<BLOCK_SIZE>(sdata, tid);
    }
    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }
}


template<int BLOCK_SIZE>
__device__ float wrapSum(float sum){
    if(BLOCK_SIZE >= 32){
        sum += __shfl_down_sync(0xffffffff, sum, 16);
    }
    if(BLOCK_SIZE >= 16){
        sum += __shfl_down_sync(0xffffffff, sum, 8);
    }
    if(BLOCK_SIZE >= 8){
        sum += __shfl_down_sync(0xffffffff, sum, 4);
    }
    if(BLOCK_SIZE >= 4){
        sum += __shfl_down_sync(0xffffffff, sum, 2);
    }
    if(BLOCK_SIZE >= 2){
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    return sum;
}
template<int BLOCK_SIZE, int NUM_PER_THREAD>
__global__ void reduce_v4(float* input, float* output, const int n){
    float sum = 0;
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * NUM_PER_THREAD) + tid;
    #pragma unroll
    for(int i = 0; i < NUM_PER_THREAD; i++){
        sum += input[idx + i * blockDim.x];
    }
    __shared__ float sdata[32];
    int laneid = tid % 32;
    int wrapid = tid / 32;
    sum = wrapSum<BLOCK_SIZE>(sum);
    if(laneid == 0){
        sdata[wrapid] = sum;
    }
    __syncthreads();
    sum = (tid < (BLOCK_SIZE / 32)) ? sdata[laneid] : 0;
    if(wrapid == 0){
        sum = wrapSum<BLOCK_SIZE / 32>(sum);
    }
    if(tid == 0){
        output[blockIdx.x] = sum;
    }
}

__device__ float wrap_reduce(float val){
    for(int i = 16; i >= 1; i >>= 1){
        val += __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}
__global__ void reduce_matrix(bfloat16* input, bfloat16* output, const int M, const int N){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row = bid;
    input = input + row * N;
    float sum = 0.f;
    for(int i = tid; i < blockDim.x; i += blockDim.x){
        float x = reinterpret_cast<float>(input[i]);
        sum += x;
    }    
    sum = wrap_reduce(sum);
    int wrap_id = tid >> 5;
    int lane_id = tid & 31;
    __shared__ float wrap_sum[32];
    if(lane_id == 0){
        wrap_sum[wrap_id] = sum;
    }
    __syncthreads();
    int num_wraps = (blockDim.x + 31) >> 5;
    if(wrap_id == 0){
        sum  = (lane_id < num_wraps) ? wrap_sum[lane_id] : 0.f;
        sum = wrap_reduce(sum);
        if(lane_id == 0){
            output[row] = (bfloat16)sum;
        }
    }
}