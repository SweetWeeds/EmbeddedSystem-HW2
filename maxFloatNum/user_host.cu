#include "user_host.cuh"

__host__
void host_maxValueVector(float *vec, int vector_size, float *p_ret_val) {
    float maxVal = FLOAT_MIN_VAL;
    for (int i = 0; i < vector_size; i++) {
        maxVal = (maxVal < vec[i]) ? vec[i] : maxVal;
    }
    *p_ret_val = maxVal;
}
