#include "user_device.cuh"

__device__ float global_cache[GLOBAL_CACHE_SIZE];
__device__ int block_cnt = 0;


/**
 * function name: atomicMax_float
 * Return Type: float
 * Description:
 *      "atomicMax" for float.
 *      Compare old value (*maxVal) and new value (value).
 *      If new value is larger than old value, than new value will overwrite old value.
 *      This function will return old value.
 */
__device__
float atomicMax_float(float *maxVal, float value) {
    float f_old = *maxVal;
    return atomicCAS((int *)maxVal, __float_as_int(f_old), __float_as_int(f_old < value ? value : f_old));
}


/**
 * function name: device_maxValueVector
 * Return Type: void
 * Description:
 *      This kernel function 
 */
__global__
void device_maxValueVector(float *vec, float *p_maxVal, int vector_size, int *p_block_cnt, int numOps) {
    extern __shared__ float cache[];
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    float tmpMax, tmpCmp;
    
    // Stage 1: Thread, Get max value of thread's part
    for (int i = 0; i < numOps; i++) {
        if (index < vector_size) {
            if (i == 0) {
                tmpMax = vec[index];
            } else {
                tmpCmp = vec[index];
                tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
            }
            index += offset;
        } else {
            if (i == 0) return;
            else break;
        }
    }
    cache[threadIdx.x] = tmpMax;
    __syncthreads();

    // Stage 2: Thread Block, Get max value of thread block's part
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            if (i == 0) {
                tmpMax = cache[i];
            } else {
                tmpCmp = cache[i];
                tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
            }
        }
        global_cache[blockIdx.x] = tmpMax;
        //atomicAdd(p_block_cnt, 1);  // Counter for synchronization of thread blocks.
        atomicAdd(&block_cnt, 1);
    } else {
        return;
    }

    // Stage 3: Grid, Get max value of from thread blocks' results.
    //if ((*p_block_cnt) == gridDim.x) {
    if (block_cnt == gridDim.x) {
        for (int i = 0; i < gridDim.x; i++) {
            if (i == 0) {
                tmpMax = global_cache[i];
            } else {
                tmpCmp = global_cache[i];
                tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
            }
        }
        (*p_maxVal) = tmpMax;
    } else {
        return;
    }
}
