__global__ void transpose(float* a, float *b, int M, int N){
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    if(tx < N && ty < M){
        b[tx * M + ty] = a[ty * N + tx];
    }
}

template<int Bm, int Bn>
__global__ void transpose_v1(float* a, float* b, int M, int N){
    int row = blockIdx.y * Bm;
    int col = blockIdx.x * Bn;
    for(int i = threadIdx.y; i < Bm; i += blockDim.y){
        int r = row + i;
        if(r < M){
            for(int j = threadIdx.x; j < Bn; j += blockDim.x){
                int c = col + j;
                if(c < N){
                    b[c * M + r] = a[r * N + c];
                }
            }
        }
    }
}

template<int Bm, int Bn>
__global__ void transpose_v2(float* a, float* b, int M, int N){
    __shared__ float tile[Bm][Bn];
    int r0 = blockIdx.y * Bm;
    int c0 = blockIdx.x * Bn;
    for(int i = threadIdx.y; i < Bm; i += blockDim.y){
        int r = r0 + i;

        if(r < M){
            for(int j = threadIdx.x; j < Bn; j += blockDim.x){
                int c = c0 + j;
                if(c < N){
                    tile[i][j] = a[r * N + c];
                }
            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.y; i < Bn; i += blockDim.y){
        int c = c0 + i;
        if(c < N){
            for(int j = threadIdx.x; j < Bm; j += blockDim.x){
                int r = r0 + j;
                if(r < M){
                    b[c * M + r] = tile[j][i];
                }
            }
        }
    }
}


