__global__ void naive_gemm(float* a, float* b, float* c, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.f;
    if(row < M && col < N){
        for(int i = 0; i < K; ++i){
            value += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = value;
    }
}
template<int BM, int BN, int BK>
__global__ void gemm_v1(float* a, float* b, float* c, int M, int N, int K){
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int load_smem_a_m = tid / BK;
    int load_smem_a_k = tid % BK;
    int load_smem_b_k = tid / BN;
    int load_smem_b_n = tid % BN;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    float sum = 0.f;
    for(int bk = 0; bk < (K + BK - 1) / BK; bk ++){
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        int load_gmem_b_k = bk * BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();
        for(int k = 0; k < BK; k++){
            sum += s_a[load_smem_a_m][k] * s_b[k][load_smem_b_n];
       }
       __syncthreads();
    }
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
}

const int BM = 64;
const int BN = 64;
const int BK = 16;
__global__ void gemm(float* A, float* B, float* C, const int M, const int N, const int K){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];
    int tid = ty * blockDim.x + tx;
    int load_smem_a_m = tid / BK;
    int load_smem_a_k = tid % BK;
    int load_smem_b_k = tid / BN;
    int load_smem_b_n = tid % BN;

    float sum = 0.f;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    int load_gmem_a_m = by * BM + load_smem_a_m;
    
    for(int bk = 0; bk < (K + BK - 1) / BK; bk++){
        
        int load_gmem_a_k = bk * BK + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
        s_A[load_smem_a_m][load_smem_a_k] = A[load_gmem_a_addr];
        int load_gmem_b_K = bk* BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_K * N + load_gmem_b_n;
        s_B[load_smem_b_k][load_smem_b_n] = B[load_gmem_b_addr];
        __syncthreads();
        
        for(int k = 0; k < BK; k++){
            sum += s_A[load_smem_a_m][k] * s_B[k][load_smem_b_n];
        }
        __syncthreads();
    }
    c[load_gmem_a_m * N + load_gmem_b_n] = sum;
}
