#include "user.cuh"

void generateVector(float *vec, int size, float *p_minVal, float *p_maxVal) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-100000.0, 100000.0);

    for (int i = 0; i < size; i++) {
        vec[i] = dis(gen);
        (*p_minVal) = (*p_minVal > vec[i]) ? vec[i] : (*p_minVal);
        (*p_maxVal) = (*p_maxVal < vec[i]) ? vec[i] : (*p_maxVal);
    }
}
