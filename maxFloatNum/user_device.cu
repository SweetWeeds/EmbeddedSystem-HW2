#include "user_device.cuh"

__device__ float global_cache[GLOBAL_CACHE_SIZE];


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
 *      This kernel function find max value from large vector.
 *      Stage 1: Find max value from partial vector which allocated to each thread.
 *      Stage 2: Find max value from thread block.
 *      Stage 3: Find max value from grid.
 */
__global__
void device_maxValueVector(float *vec, float *p_maxVal, int vector_size, int *p_block_cnt, int numOps) {
    extern __shared__ float cache[];
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    float tmpMax = -INFINITY, tmpCmp;
    
    // Stage 1: Thread. Get max value of thread's partial vector.
    for (int i = 0; i < numOps; i++) {
        if (index < vector_size) {
            tmpCmp = vec[index];
        } else {    // index is out of range.
            tmpCmp = -INFINITY;
        }
        tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
        index += offset;
    }
    cache[threadIdx.x] = tmpMax;
    __syncthreads();

    // Stage 2: Thread Block. Get max value from 'global_cache[]'
    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            tmpCmp = cache[i];
            tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
        }
        global_cache[blockIdx.x] = tmpMax;
        atomicAdd(p_block_cnt, 1);  // Counter for synchronization of thread blocks.
    } else {
        return;
    }

    // Stage 3: Grid, Get max value of from thread blocks' results.
    if ((*p_block_cnt) == gridDim.x) {
        for (int i = 0; i < gridDim.x; i++) {
            tmpCmp = global_cache[i];
            tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
        }
        (*p_maxVal) = tmpMax;
    } else {
        return;
    }
}
