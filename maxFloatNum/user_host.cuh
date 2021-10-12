#ifndef USER_HOST_H
#define USER_HOST_H

#include "user.cuh"

__host__
void host_maxValueVector(float *vec, int vector_size, float *p_ret_val);

#endif