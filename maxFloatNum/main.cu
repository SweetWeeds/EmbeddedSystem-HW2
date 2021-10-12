#include "user.cuh"

using namespace std;

int main(int argc, char *argv[]) {
    string fileName;
    if (argc == 1) fileName = "maxFloatNum.csv";
    else fileName = argv[1];
    CSV_Data cd(fileName, true);

    // Generate Vector
    printf("[INFO] Generating vector...\n");
    float minVal = INFINITY, maxVal = -INFINITY;
    float *p_host_vector = (float *)malloc(VECTOR_SIZE * sizeof(float));
    generateVector(p_host_vector, VECTOR_SIZE, &minVal, &maxVal);
    printf("[INFO] Max value:%.4f, Min Value:%.4f\n", maxVal, minVal);

    // Copy vec from host to device
    float *p_host_max_val   = (float *)malloc(sizeof(float)), *p_device_vector;
    float *p_host_device_max_val = (float *)malloc(sizeof(float)), *p_device_max_val;
    cudaMalloc((void **)&p_device_vector, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void **)&p_device_max_val, sizeof(float));
    cudaMemcpy(p_device_vector, p_host_vector, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    printf("Starting Benchmark (ITERATION: %d)\n", ITERATION);

    // Host Code
    struct timeval startTime, endTime;
    float host_exec_time = 0.0;
    for (int iter = 0; iter < ITERATION; iter++) {
        gettimeofday(&startTime, NULL);
        host_maxValueVector(p_host_vector, VECTOR_SIZE, p_host_max_val);
        gettimeofday(&endTime, NULL);
        float tmp_exec_time = (endTime.tv_sec - startTime.tv_sec) * 1000. + (endTime.tv_usec - startTime.tv_usec) / 1000.;
        host_exec_time += tmp_exec_time;
    }
    host_exec_time /= ITERATION;
    cd.AddData("host", 1, 1, host_exec_time);
    //printf("[INFO] Host Execution time: %.4f\n", host_exec_time);

    // Device Code
    for (int numThreadsPerBlk = NUM_THREADS_BASE; numThreadsPerBlk <= NUM_THREADS_MAX; numThreadsPerBlk*=2) {
        for (int numBlks = NUM_THREAD_BLKS_FROM; numBlks <= NUM_THREAD_BLKS_TO; numBlks*=2) {
            int numElements = VECTOR_SIZE;
            int numOps = numElements > (numBlks * numThreadsPerBlk) ?
                        numElements / (numBlks * numThreadsPerBlk) + (numElements % (numBlks * numThreadsPerBlk) ? 1 : 0) : 1;
            int *p_device_block_cnt, host_block_cnt;
            cudaMalloc((void **)&p_device_block_cnt, sizeof(int));
            dim3 gridSize(numBlks);
            dim3 blockSize(numThreadsPerBlk);
            float avg_exec_time = 0.0;
            for (int iter = 0; iter < ITERATION; iter++) {
                cudaEvent_t cuda_start, cuda_end;
                float tmp_exec_time;
                cudaEventCreate(&cuda_start);
                cudaEventCreate(&cuda_end);

                // Prepare data
                cudaMemset(p_device_max_val, 0, sizeof(float));
                cudaMemset(p_device_block_cnt, 0, sizeof(int));
                
                // Execute Kernel
                cudaEventRecord(cuda_start, 0);
                device_maxValueVector<<<gridSize, blockSize, numThreadsPerBlk*sizeof(float)>>>
                                    (p_device_vector, p_device_max_val, VECTOR_SIZE, p_device_block_cnt, numOps);
                cudaDeviceSynchronize();
                cudaEventRecord(cuda_end, 0);
                cudaEventSynchronize(cuda_end);
                cudaEventElapsedTime(&tmp_exec_time, cuda_start, cuda_end);
                cudaEventDestroy(cuda_start);
                cudaEventDestroy(cuda_end);
                avg_exec_time += tmp_exec_time;
            }

            avg_exec_time /= ITERATION; // Get average of execution time.

            // Print Results
            cudaMemcpy(&host_block_cnt, p_device_block_cnt, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(p_host_device_max_val, p_device_max_val, sizeof(float), cudaMemcpyDeviceToHost);
            cd.AddData("device", numThreadsPerBlk, numBlks, avg_exec_time);
            //if (*p_host_device_max_val != *p_host_max_val) {
            //    printf("   [ERROR] Values are not matching.\n"\
            //           "           host: %.4f"\
            //           "           device: %.4f", *p_host_max_val, *p_host_device_max_val);
            //}

            cudaFree(p_device_block_cnt);
        }
    }

    // Deallocation
    cudaFree(p_device_vector);
    cudaFree(p_device_max_val);
    free(p_host_vector);
    free(p_host_max_val);
    free(p_host_device_max_val);
}
