__global__ void naive_gemm(float *A, float *B, float *C, const int M, const int N, const int K){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if(row < M && col < N){
        float value = 0.f;
        for(int k = 0; k < K; k++){
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}


// one block for one tile of C
template<int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8,
        int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void block_tile_GEMM(float *A, float *B, float *C, const int M, const int N, const int K){
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

    // this block begin iter of left-top element
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;

    // one dimension of block
    int tid = threadIdx.x;
    
    constexpr int A_BLOCK_Y = blockSize / A_BLOCK_X;

    // for the thread named `tid` in the block, the pos is `(tid / A_BLOCK_X, tid % A_BLOCK_X)`
    int A_THREAD_Y = tid / A_BLOCK_X;
    int A_THREAD_X = tid % A_BLOCK_X;


    // same for B
    constexpr int B_BLOCK_Y = blockSize / B_BLOCK_X;
    int B_THREAD_Y = tid / B_BLOCK_X;
    int B_THREAD_X = tid % B_BLOCK_X;

    constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;
    int C_THREAD_Y = tid / C_BLOCK_X;
    int C_THREAD_X = tid % C_BLOCK_X;

    // Tm * Tn elements for each thread
    constexpr int Tm = Bm / C_BLOCK_Y;
    constexpr int Tn = Bn / C_BLOCK_X;

    float Ct[Tm][TN] = {0.f};

    for(int k = 0; k < K; k += Bk){
        #pragma unroll
        for(int i = A_THREAD_Y; i < Bm; i += A_BLOCK_Y){
            int r = r0 + i;
            #pragma unroll
            for(int j = A_THREAD_X; j < Bk; j += A_BLOCK_X){
                int c = k + j;
                As[i][j] = (r < M && c < K) ? A[r * K + c] : 0.f;
            }
        }

        #pragma unroll
        for (int i = B_THREAD_Y; i < Bk; i += B_BLOCK_Y) {
            int r = k + i;
            #pragma unroll
            for (int j = B_THREAD_X; j < Bn; j += B_BLOCK_X) {
                int c = c0 + j;
                Bs[i][j] = (r < K && c < N) ? B[r * N + c] : 0.f;
            }
        }
        __syncthreads();

        #pragma unroll
        for(int p = 0; p < Bk; p++){
            #pragma unroll
            for(int i = 0; i < Tm; i++){
                int r = C_THREAD_Y + i * C_BLOCK_Y;
                #pragma unroll
                for(int j = 0; j < Tn; j++){
                    int c = C_THREAD_X + j * C_BLOCK_X;
                    Ct[i][j] += As[r][p] * Bs[p][c];
                }
            }
        }
        __syncthreads();
    }
    // 将 Ct 写入 C
    #pragma unroll
    for (int i = 0; i < Tm; ++i){
        int r = r0 + C_THREAD_Y + i * C_BLOCK_Y;
        #pragma unroll
        for (int j = 0; j < Tn; ++j){
            int c = c0 + C_THREAD_X + j * C_BLOCK_X;
            if (r < M && c < N){ 
                C[r * N + c] = Ct[i][j];
            }
    }
  }
}