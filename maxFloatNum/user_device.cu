#include "user_device.h"

//__shared__ float global_cache[GLOBAL_CACHE_SIZE];
__device__ float global_cache[GLOBAL_CACHE_SIZE];

__device__
float atomicMax_float(float *maxVal, float value) {
    float f_old = *maxVal;
    return atomicCAS((int *)maxVal, __float_as_int(f_old), __float_as_int(f_old < value ? value : f_old));
}

__global__
void device_parallelized_maxValueVector(float *vector, float *p_maxVal, int vector_size, int *p_block_cnt, int numOps, int *p_device_check_vector) {
    extern __shared__ float cache[];
    int base_index = (blockDim.x * blockIdx.x + threadIdx.x) * numOps;
    float tmpMax, tmpCmp;
    
    // Stage 1: Thread, Get max value of thread's part
    vector = &vector[base_index];
    tmpMax = vector[0];
    atomicAdd(p_block_cnt, 1);
    for (int i = 1; i < numOps; i++) {
        if ((base_index + i) < vector_size) {
            tmpCmp = vector[i];
            tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
            atomicAdd(p_block_cnt, 1);
        }
    }
    cache[threadIdx.x] = tmpMax;
    __syncthreads();

    // Stage 2: Thread Block, Get max value of thread block's part
    if (threadIdx.x == 0) {
        tmpMax = cache[0];
        for (int i = 1; i < blockDim.x; i++) {
            tmpMax = (tmpMax < cache[i]) ? cache[i] : tmpMax;
        }
        global_cache[blockIdx.x] = tmpMax;
    } else {
        return;
    }

    // Stage 3: Grid, Get max value of all of vector
    if (blockIdx.x == 0) {
        tmpMax = global_cache[0];
        for (int i = 1; i < gridDim.x; i++) {
            tmpCmp = global_cache[i];
            tmpMax = (tmpMax < tmpCmp) ? tmpCmp : tmpMax;
        }
        (*p_maxVal) = tmpMax;
    } else {
        return;
    }
}


__global__
void device_simple_maxValueVector(float *vector, int vector_size, int numOps, float *p_maxVal) {
    int base_index = (blockDim.x * blockIdx.x + threadIdx.x) * numOps;
    //int offset = gridDim.x * blockDim.x;

    for (int i = 0; i < numOps; i++) {
        if ((base_index + i) < vector_size) {
            atomicMax_float(p_maxVal, vector[base_index + i]);
        } else {
            return;
        }
    }
}
