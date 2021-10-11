#include "user_device.h"

//__shared__ float global_cache[GLOBAL_CACHE_SIZE];
__device__ float global_cache[GLOBAL_CACHE_SIZE];

__device__
float atomicMax_float(float *maxVal, float value) {
    float f_old = *maxVal;
    return atomicCAS((int *)maxVal, __float_as_int(f_old), __float_as_int(f_old < value ? value : f_old));
}

__global__
void device_parallelized_maxValueVector(float *vector, float *p_maxVal, int vector_size, int *p_block_cnt, int numOps) {
    extern __shared__ float cache[];
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    float tmp;

    // Stage 1: Thread, Get max value of thread's part
    for (int i = 0; i < numOps; i++) {
        if (i == 0) {
            cache[threadIdx.x] = vector[index];
        } else {
            if (index < vector_size) {
                atomicMax_float(&cache[threadIdx.x], vector[index]);
            }
        }
        index += offset;
    }
    __syncthreads();

    // Stage 2: Thread Block, Get max value of thread block's part
    if (threadIdx.x == 0) {
        tmp = cache[0];
        for (int i = 1; i < blockDim.x; i++) {
            //atomicMax_float(&tmp, cache[i]);
            tmp = (tmp < cache[i]) ? cache[i] : tmp;
        }
        global_cache[blockIdx.x] = tmp;
        (*p_block_cnt) = (*p_block_cnt) + 1;
    } else {
        return;
    }

    // Stage 3: Grid, Get max value of all of vector
    //if (blockIdx.x == 0) {
    if ((*p_block_cnt) == gridDim.x) {
        tmp = global_cache[0];
        for (int i = 1; i < gridDim.x; i++) {
            //atomicMax_float(&tmp, global_cache[i]);
            tmp = (tmp < global_cache[i]) ? global_cache[i] : tmp;
        }
        *p_maxVal = tmp;
    }
}


__global__
void device_simple_maxValueVector(float *vector, int vector_size, int numOps, float *p_maxVal) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int i = 0; i < numOps; i++) {
        if (index < vector_size) {
            atomicMax_float(p_maxVal, vector[index]);
        }
        index += offset;
    }
}
