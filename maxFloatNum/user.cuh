#ifndef USER_H
#define USER_H

#include <iostream>
#include <random>
#include <sys/time.h>
#include <cmath>

#include "user_device.cuh"
#include "user_host.cuh"
#include "csv_data.cuh"

#define FLOAT_MIN_VAL           1.175494351E-38
#define VECTOR_SIZE             100000000   // Needs 95 MB for float vector (a billion)

void generateVector(float *vec, int size, float *p_minVal, float *p_maxVal);

#endif
