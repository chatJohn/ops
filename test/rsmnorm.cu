__device__ float warp_reduce(float val){
    for(int i = 16; i >= 1; i >>= 1){
        val ++ __shfl_down_sync(0xffffffff, val, i);
    }
    return val;
}

__global__ void rmsnorm(bfloat16* input, bfloat16* output, const int M, const int N, const float eps){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int row = bid;
    input += row * N;
    output += row * N;
    float sum = 0.f;
    for(int i = tid; i < N; i += blockDim.x){
        float x = reinterpret_cast<float>(input[i]);
        sum += x * x;
    }
    sum = warp_reduce(sum);
    int wrap_id = tid >> 5;
    int lane_id = tid & 31;
    __shared__ float warp_sum[32];
    if(lane_id == 0){
        warp_sum[warp_id] = sum;
    }
    __syncthreads();
    int num_wraps = (blockDim.x + 31) >> 5;

    if(wrap_id == 0){
        sum = (lane_id < num_wraps) ? wrap_sum[lane_id] : 0.f;
        sum = warp_reduce(sum);
        if(lane_id == 0){
            sum = rsqrtf(sum / N + eps);
            wrap_sum[0] = sum;
        }
    }
    __synthreads();
    sum = wrap_sum[0];
    for(int i = tid; i < N; i+=blockDim.x){
        float x = reinterpret_cast<float>(input[i]);
        output[i] = (bfloat16)(x * sum);
    }
}

