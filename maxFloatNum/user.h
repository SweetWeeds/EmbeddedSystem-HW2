#ifndef USER_H
#define USER_H

#include <iostream>
#include <random>
#include <sys/time.h>
#include <cmath>

#define FLOAT_MIN_VAL           1.175494351E-38
#define VECTOR_SIZE             100000000   // Needs 95 MB for float vector
#define NUM_THREADS_BASE        32
#define NUM_THREADS_MAX         1024
#define NUM_THREAD_BLKS_FROM    1
#define NUM_THREAD_BLKS_TO      512
#define ITERATION               100

#endif