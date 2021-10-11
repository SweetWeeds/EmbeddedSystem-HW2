#include "user_host.h"

__host__
void host_maxValueVector(float *vector, int vector_size, float *p_ret_val) {
    float maxVal = FLOAT_MIN_VAL;
    for (int i = 0; i < vector_size; i++) {
        maxVal = (maxVal < vector[i]) ? vector[i] : maxVal;
    }
    *p_ret_val = maxVal;
}
