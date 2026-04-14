#include "common.h"
#include <cuda/atomic>

__global__ void kernel0(unsigned int n, unsigned int k, unsigned int* rowPtrs, unsigned int* colIdxs, float* vals, float* B, float* X, int* ready, int* nextRow){
    unsigned int b = threadIdx.x; //RHS column
    __shared__ int sharedRow;
    while(true){
        if (b == 0){ //thread 0 claims the next row for the whole block fr
            cuda::atomic_ref<int, cuda::thread_scope_device> nxt(*nextRow);
            sharedRow = nxt.fetch_add(1, cuda::memory_order_relaxed);
        }
        __syncthreads();

        int i = sharedRow;
        if (i>=(int)n) return;
        if (b<k){
            float sum  = B[i*k + b];
            float diag = 1.0f;

            for (unsigned int p = rowPtrs[i]; p<rowPtrs[i+1]; ++p){
                unsigned int j = colIdxs[p];
                float val = vals[p];
                if (j<i){ //basically spin till dependency j is done
                    cuda::atomic_ref<int, cuda::thread_scope_device> ready_ref(ready[j]);
                    while (ready_ref.load(cuda::memory_order_acquire) == 0);
                    sum -= val*X[j*k + b];
                } 
                else if (j == i){
                    diag = (val != 0.0f) ? val : 1.0f;
                }
            }
            X[i*k + b] = sum/diag;
        }
        __syncthreads();

        if (b == 0){ //thread 0 => row i is done
            __threadfence(); //https://www.lrde.epita.fr/~carlinet/cours/GPGPU/j2-part4-kernel-programming.slides.pdf
            cuda::atomic_ref<int, cuda::thread_scope_device> rdy(ready[i]);
            rdy.store(1, cuda::memory_order_release);
        }
        __syncthreads();
    }
}

void sptrsv_gpu0(CSCMatrix* L_c, CSRMatrix* L_r, DenseMatrix* B, DenseMatrix* X, CSCMatrix* L_c_host, CSRMatrix* L_r_host, unsigned int numCols){
    unsigned int n = L_r_host->numRows;
    unsigned int k = numCols;

    CSRMatrix csrPtr;
    cudaMemcpy(&csrPtr, L_r, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    DenseMatrix bPtr, xPtr;
    cudaMemcpy(&bPtr, B, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&xPtr, X, sizeof(DenseMatrix), cudaMemcpyDeviceToHost);

    int* ready_d;
    int* nextRow_d;
    cudaMalloc((void**)&ready_d, n*sizeof(int));
    cudaMalloc((void**)&nextRow_d, sizeof(int));
    cudaMemset(ready_d, 0, n*sizeof(int));
    cudaMemset(nextRow_d, 0, sizeof(int));

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    dim3 block(k);
    dim3 grid(numSMs);

    kernel0<<<grid, block>>>(n, k, csrPtr.rowPtrs, csrPtr.colIdxs, csrPtr.values, bPtr.values, xPtr.values, ready_d, nextRow_d);

    cudaFree(ready_d);
    cudaFree(nextRow_d);
}
