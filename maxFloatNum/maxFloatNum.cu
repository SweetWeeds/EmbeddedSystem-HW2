#include "user.h"

using namespace std;

float *generateVector(int size, float *p_minVal, float *p_maxVal) {
    //random_device rd;
    //mt19937 gen(rd());
    //uniform_real_distribution<float> dis(0, 123456.789);

    float *vector = (float *)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        //vector[i] = dis(gen);
        //vector[i] = dis(gen);
        vector[i] = i;
        (*p_minVal) = (*p_minVal > vector[i]) ? vector[i] : (*p_minVal);
        (*p_maxVal) = (*p_maxVal < vector[i]) ? vector[i] : (*p_maxVal);
    }

    return vector;
}

int main(int argc, char *argv[]) {
    // Generate Vector
    printf("[INFO] Generating vector...\n");
    float minVal = INFINITY, maxVal = -INFINITY;
    float *p_host_vector = generateVector(VECTOR_SIZE, &minVal, &maxVal);
    printf("[INFO] Max value:%.4f, Min Value:%.4f\n", maxVal, minVal);

    // Copy vector from host to device
    float *p_host_max_val   = (float *)malloc(sizeof(float)), *p_device_vector;
    float *p_host_device_max_val = (float *)malloc(sizeof(float)), *p_device_max_val;
    int *p_host_check_vector = (int *)malloc(VECTOR_SIZE * sizeof(int)), *p_device_check_vector;
    cudaMalloc((void **)&p_device_vector, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void **)&p_device_max_val, sizeof(float));
    cudaMalloc((void **)&p_device_check_vector, VECTOR_SIZE * sizeof(int));
    cudaMemcpy(p_device_vector, p_host_vector, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Host Code
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    host_maxValueVector(p_host_vector, VECTOR_SIZE, p_host_max_val);
    gettimeofday(&endTime, NULL);
    float host_exec_time = (endTime.tv_sec - startTime.tv_sec) * 1000. + (endTime.tv_usec - startTime.tv_usec) / 1000.;
    printf("[INFO] Host Execution time: %.4f\n", host_exec_time);

    // Device Code
    for (int numThreadsPerBlk = NUM_THREADS_BASE; numThreadsPerBlk <= NUM_THREADS_MAX; numThreadsPerBlk*=2) {
    //for (int numThreadsPerBlk = NUM_THREADS_BASE; numThreadsPerBlk <= NUM_THREADS_BASE; numThreadsPerBlk*=2) {
        for (int numBlks = NUM_THREAD_BLKS_FROM; numBlks <= NUM_THREAD_BLKS_TO; numBlks*=2) {
        //for (int numBlks = NUM_THREAD_BLKS_FROM; numBlks <= NUM_THREAD_BLKS_FROM; numBlks*=2) {
            //int numOps = int(ceil(VECTOR_SIZE / float(numThreadsPerBlk*numBlks)));
            int numElements = VECTOR_SIZE;
            int numOps = numElements > (numBlks * numThreadsPerBlk) ?
                        numElements / (numBlks * numThreadsPerBlk) + (numElements % (numBlks * numThreadsPerBlk) ? 1 : 0) : 1;
            int *p_device_block_cnt, host_block_cnt;
            cudaMalloc((void **)&p_device_block_cnt, sizeof(int));
            dim3 gridSize(numBlks);
            dim3 blockSize(numThreadsPerBlk);

            cudaEvent_t cuda_start, cuda_end;
            float tmp_exec_time;
            cudaEventCreate(&cuda_start);
            cudaEventCreate(&cuda_end);

            // Prepare data
            cudaMemset(p_device_max_val, 0, sizeof(float));
            cudaMemset(p_device_block_cnt, 0, sizeof(int));
            cudaMemset(p_device_check_vector, 0, VECTOR_SIZE * sizeof(int));
            
            //device_maxValueVector<<<gridSize, blockSize>>>(p_device_vector, VECTOR_SIZE, elementsPerThread);
            cudaEventRecord(cuda_start, 0);
            //device_simple_maxValueVector<<<gridSize, blockSize>>>(p_device_vector, VECTOR_SIZE, numOps, p_device_max_val);
            device_parallelized_maxValueVector<<<gridSize, blockSize, numThreadsPerBlk*sizeof(float)>>>
                                (p_device_vector, p_device_max_val, VECTOR_SIZE, p_device_block_cnt, numOps, p_device_check_vector);
            cudaDeviceSynchronize();
            cudaEventRecord(cuda_end, 0);
            cudaEventSynchronize(cuda_end);
            cudaEventElapsedTime(&tmp_exec_time, cuda_start, cuda_end);
            cudaEventDestroy(cuda_start);
            cudaEventDestroy(cuda_end);

            // Print Results
            cudaMemcpy(&host_block_cnt, p_device_block_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_host_device_max_val, p_device_max_val, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_host_check_vector, p_device_check_vector, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
            if (*p_host_device_max_val == *p_host_max_val) {
                printf("[INFO] ");
            } else {
                printf("[WARN] ");
            }
            printf("numOps: %10d, numThreadsPerBlk: %5d, numBlks: %5d, host result:%10.4f, device result:%15.4f, block_cnt:%5d, exec_time:%.4f\n",
                    numOps, numThreadsPerBlk, numBlks, *p_host_max_val, *p_host_device_max_val, host_block_cnt, tmp_exec_time);
            //int cnt = 0, minIdx = VECTOR_SIZE, maxIdx = 0;
            //for (int i = 0; i < VECTOR_SIZE; i++) {
            //    if (p_host_check_vector[i] == 0) {
            //        cnt++;
            //        minIdx = minIdx > i ? i : minIdx;
            //        maxIdx = maxIdx < i ? i : maxIdx;
            //    }
            //}
            //printf("    Not indexed cnt:%d, minIdx:%d, maxIdx:%d\n", cnt, minIdx, maxIdx);
            cudaFree(p_device_block_cnt);
        }
    }

    // Deallocation
    cudaFree(p_device_vector);
    cudaFree(p_device_max_val);
    cudaFree(p_device_check_vector);
    free(p_host_vector);
    free(p_host_max_val);
    free(p_host_device_max_val);
    free(p_host_check_vector);
}
