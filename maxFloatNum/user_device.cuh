#ifndef USER_DEVICE_H
#define USER_DEVICE_H

#include "user.cuh"

#define GLOBAL_CACHE_SIZE 512

#define NUM_THREADS_BASE        32
#define NUM_THREADS_MAX         1024
#define NUM_THREAD_BLKS_FROM    1
#define NUM_THREAD_BLKS_TO      512
#define ITERATION               10

__global__
void device_maxValueVector(float *vec, float *p_maxVal, int vector_size, int *p_block_cnt, int numOps);

#endif