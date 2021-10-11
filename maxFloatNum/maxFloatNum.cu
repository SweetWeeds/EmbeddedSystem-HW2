#include "user.h"
#include "user_device.h"
#include "user_host.h"

using namespace std;

int host_block_cnt = 0;
int *p_device_block_cnt;

float *generateVector(int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-123.45, 123.45);

    float *vector = (float *)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        //vector[i] = dis(gen);
        vector[i] = float(i * 100) / dis(gen);
    }

    return vector;
}

int main(int argc, char *argv[]) {
    // Memory Allocation
    printf("[INFO] Generating vector...\n");
    float minVal = -INFINITY, maxVal = INFINITY;
    float *host_p_vector = generateVector(VECTOR_SIZE);
    printf("[INFO] Max value:%.4f, Min Value:%.4f\n", maxVal, minVal);
    float *p_device_vector, *p_device_max_val;
    float *p_host_max_val   = (float *)malloc(sizeof(float));
    float *p_host_device_max_val = (float *)malloc(sizeof(float));
    cudaMalloc((void **)&p_device_vector, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void **)&p_device_max_val, sizeof(float));
    cudaMalloc((void **)&p_device_block_cnt, sizeof(int));
    cudaMemcpy(p_device_vector, host_p_vector, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Host Code
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    host_maxValueVector(host_p_vector, VECTOR_SIZE, p_host_max_val);
    gettimeofday(&endTime, NULL);
    float host_exec_time = (endTime.tv_sec - startTime.tv_sec) * 1000. + (endTime.tv_usec - startTime.tv_usec) / 1000.;
    printf("[INFO] Host Execution time: %.4f\n", host_exec_time);

    // Device Code
    for (int numThreadsPerBlk = NUM_THREADS_BASE; numThreadsPerBlk <= NUM_THREADS_MAX; numThreadsPerBlk*=2) {
        for (int numBlks = NUM_THREAD_BLKS_FROM; numBlks <= NUM_THREAD_BLKS_TO; numBlks*=2) {
            int numOps = VECTOR_SIZE / (numThreadsPerBlk*numBlks);
            dim3 gridSize(numBlks);
            dim3 blockSize(numThreadsPerBlk);

            cudaEvent_t cuda_start, cuda_end;
            float tmp_exec_time;
            cudaEventCreate(&cuda_start);
            cudaEventCreate(&cuda_end);

            // Prepare data
            host_block_cnt = 0;
            cudaMemcpy(p_device_block_cnt, &host_block_cnt, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(p_device_max_val, &minVal, sizeof(float), cudaMemcpyHostToDevice);

            //device_maxValueVector<<<gridSize, blockSize>>>(p_device_vector, VECTOR_SIZE, elementsPerThread);
            cudaEventRecord(cuda_start, 0);
            //device_simple_maxValueVector<<<gridSize, blockSize>>>(p_device_vector, VECTOR_SIZE, numOps, p_device_max_val);
            device_parallelized_maxValueVector<<<gridSize, blockSize, blockSize.x>>>(p_device_vector, p_device_max_val, VECTOR_SIZE, p_device_block_cnt, numOps);
            cudaEventRecord(cuda_end, 0);
            cudaEventSynchronize(cuda_end);
            cudaEventElapsedTime(&tmp_exec_time, cuda_start, cuda_end);
            cudaEventDestroy(cuda_start);
            cudaEventDestroy(cuda_end);

            cudaMemcpy(&host_block_cnt, p_device_block_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_host_device_max_val, p_device_max_val, sizeof(float), cudaMemcpyDeviceToHost);
            if (*p_host_device_max_val == *p_host_max_val) {
                printf("[INFO] ");
            } else {
                printf("[WARN] ");
            }
            printf("numOps: %d, numThreadsPerBlk: %d, numBlks: %d, host result:%.4f, device result:%.4f, block_cnt:%d, exec_time:%.4f\n",
                    numOps, numThreadsPerBlk, numBlks, *p_host_max_val, *p_host_device_max_val, host_block_cnt, tmp_exec_time);
        }
    }

    // Deallocation
    cudaFree(p_device_vector);
    cudaFree(p_device_max_val);
}
