
// Naive SGEMM: A * B = C
// element per thread, row major
__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, const int M, const int N, const int K){
    int x = blockIdx.x * blockDim.x + threadIdx.x; // global x of col
    int y = blockIdx.y * blockDim.y + threadIdx.y; // global y of row
    if(x < N && y < M){
        float psum = 0.0;
        #pragma unroll
        for(int k = 0; k < K; k++){
            psum += a[y * K + k] * b[k * N + x];
        }
        c[y * N + x] = psum;
    }
}

// Compute a tile of C, one block per tile
// BM is row size of tile in C, BN is col size, BK is K_loop size.
template <int BM = 32, int BN = 32, int BK = 32>
__global__ void sgemm_tile_kloop_f32_kernel(float *a, float *b, float *c, const int M, const int N, const int K){
    // all threads in a block share the shared memory of this block
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];
    int bx = blockIdx.x; // col
    int by = blockIdx.y; // row
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // not global tid, just local tid in a  block
    // load data from global memory to shared memory, one thread in a block loads a element
    // in this case, there are 32 * 32 threads in a block, and BM = BK = BN = 32
    int load_smem_a_m = tid / BK; // one ele in global move to where in a_shared
    int load_smem_a_k = tid % BK;
    int load_smem_b_k = tid / BN; // same for b_shared
    int load_smem_b_n = tid % BN;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n; // contrast global idx to shared idx
    float sum = 0.f;
    for


}