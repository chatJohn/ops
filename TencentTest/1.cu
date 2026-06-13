/*
    2048个元素的求和
*/

__device__ float wrap_sum(float sum){
    for(int i = 16; i >= 1; i >>= 1){
        sum ++ __shfl_down_sync(0xffffffff, sum, i);
    }
    return sum;
}
__global__ void reduce(float* A, float* B, const int N){
   int bid = blockIdx.x;
   int tid = threadIdx.x;
   float sum = 0.f;
   for(int i = tid; i < N; i += blockDim.x){
    sum += A[i];
   } 
   sum = wrap_sum(sum);
   int wrap_id = tid >> 5;
   int lane_id = tid & 31;
   __shared__ float wrap_sum[32];
   if(lane_id == 0){
    wrap_sum[wrap_id] = sum;
   }
   __syncthreads();
   if(wrap_id == 0){
    sum = (lane_id < (blockDim.x + 31) >> 5) ? wrap_sum[lane_id] : 0.f;
    sum = wrap_sum(sum);
    if(tid == 0){
        B[bid] = sum;
    }
   }

}