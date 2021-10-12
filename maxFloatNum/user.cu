#include "user.cuh"

void generateVector(float *vec, int size, float *p_minVal, float *p_maxVal) {
    for (int i = 0; i < size; i++) {
        vec[i] = i;
        (*p_minVal) = (*p_minVal > vec[i]) ? vec[i] : (*p_minVal);
        (*p_maxVal) = (*p_maxVal < vec[i]) ? vec[i] : (*p_maxVal);
    }
}
