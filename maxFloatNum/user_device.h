#ifndef USER_DEVICE_H
#define USER_DEVICE_H

#include "user.h"

#define GLOBAL_CACHE_SIZE 512

__global__
void device_parallelized_maxValueVector(float *vector, float *p_maxVal, int vector_size, int *p_block_cnt, int numOps, int *p_device_check_vector);

__global__
void device_simple_maxValueVector(float *vector, int vector_size, int numOps, float *maxVal);

#endif