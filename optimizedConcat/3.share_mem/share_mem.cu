#include "user.h"

using namespace std;

int main(int argc, char *argv[]) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(1, 10);
    cudaEvent_t cuda_start, cuda_end;
    float host_exec_time, device_exec_time;

    printf("mat1[%d][%d]\n", MAT1_ROW, MAT1_COL);
    printf("mat2[%d][%d]\n", MAT2_ROW, MAT2_COL);
    printf("matr[%d][%d]\n", MATR_ROW, MATR_COL);

    int mat1_size = MAT1_ROW * MAT1_COL * sizeof(int);
    int mat2_size = MAT2_ROW * MAT2_COL * sizeof(int);
    int matr_size = MATR_ROW * MATR_COL * sizeof(int);
    
    int *host_mat1 = NULL, *host_mat2 = NULL, *host_matr = NULL;
    int *device_mat1 = NULL, *device_mat2 = NULL, *device_matr = NULL;
    int *host_device_matr = NULL;
    
    // Start of Memory Allocation //
    host_mat1 = (int *)malloc(mat1_size);
    host_mat2 = (int *)malloc(mat2_size);
    host_matr = (int *)malloc(matr_size);
    host_device_matr = (int *)malloc(matr_size);
    cudaMalloc((void **)&device_mat1, mat1_size);
    cudaMalloc((void **)&device_mat2, mat2_size);
    cudaMalloc((void **)&device_matr, matr_size);
    // End of Memory Allocation //

    // Start of Array Initalization //
    // Matrix 1
    for (int i = 0; i < MAT1_ROW; i++) {
        for (int j = 0; j < MAT1_COL; j++) {
            host_mat1[i * MAT1_COL + j] = dis(gen);
        }
    }

    // Matrix 2
    for (int i = 0; i < MAT2_ROW; i++) {
        for (int j = 0; j < MAT2_COL; j++) {
            host_mat2[i * MAT2_COL + j] = dis(gen);
        }
    }
    // End of Array Initalization //

    // Memory Copy (Matrix 1, 2)
    cudaMemcpy(device_mat1, host_mat1, mat1_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mat2, host_mat2, mat2_size, cudaMemcpyHostToDevice);

    // Prepare File Stream
    ofstream resultFile;
    resultFile.open("./result.csv", ios::out);
    resultFile << "Target,#Threads,#ThreadBlks,ExecTime\n";

    // Start of Concatenation (host) //
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    host_Concatenate(host_mat1, host_mat2, host_matr, &host_exec_time);
    gettimeofday(&endTime, NULL);
    host_exec_time = (endTime.tv_sec - startTime.tv_sec) * 1000. + (endTime.tv_usec - startTime.tv_usec) / 1000.;
    resultFile << "host,1,1," << host_exec_time << "\n";    // Write information of execution on host.
    printf("[INFO] Host Execution time:%lf\n", host_exec_time);
    // End of Concatenation (host) //

    // Start of Concatenation (device) //
    int numElements = MATR_COL * MATR_ROW;
    int numBlocks, numThreadsperBlock;
    for (numThreadsperBlock = NUM_THREADS_BASE; numThreadsperBlock <= NUM_THREADSA_MAX; numThreadsperBlock *= 2) {
        for (numBlocks = NUM_THREAD_BLKS_FROM; numBlocks <= NUM_THREAD_BLKS_TO; numBlocks *= 2) {
            int numOps = numElements > (numBlocks * numThreadsperBlock) ?
                        numElements / (numBlocks * numThreadsperBlock) + (numElements % (numBlocks * numThreadsperBlock) ? 1 : 0) : 1;
            dim3 gridSize(numBlocks);
            dim3 blockSize(numThreadsperBlock);
            float tmp_exec_time;
            device_exec_time = 0.0;

            for (int i = 0; i < ITERATION; i++) {
                cudaMemset(device_matr, 0, numElements);    // Initialize values of 'device_matr'
                cudaEventCreate(&cuda_start);
                cudaEventCreate(&cuda_end);
                cudaEventRecord(cuda_start, 0);
                device_Concatenate<<<gridSize, blockSize, numThreadsperBlock>>>(device_mat1, device_mat2, device_matr,
                                                            numOps, numElements, MAT1_COL, MAT2_COL, MATR_COL);
                cudaEventRecord(cuda_end, 0);
                cudaEventSynchronize(cuda_end);
                cudaEventElapsedTime(&tmp_exec_time, cuda_start, cuda_end);
                cudaEventDestroy(cuda_start);
                cudaEventDestroy(cuda_end);
                device_exec_time += tmp_exec_time;
            }
            device_exec_time /= ITERATION;

            // Write information of execution on device.
            resultFile << "device," << numThreadsperBlock << "," << numBlocks << "," << device_exec_time << "\n";
            cudaMemcpy(host_device_matr, device_matr, matr_size, cudaMemcpyDeviceToHost);
            // Compare concatenation results of host and device
            int diff = compareArray(host_matr, host_device_matr, numElements);
            if (diff) {
                printf("[WARNING] ");   // Different points are exist.
            } else {
                printf("[INFO] ");      //
            }
            printf("numOps: %d, numBlocks: %d, numThreadsperBlock: %d, diff: %d, exec_time: %.4lf\n", numOps, numBlocks, numThreadsperBlock, diff, device_exec_time);
        }
    }
    // End of Concatenation (device) //

    free(host_mat1);
    free(host_mat2);
    free(host_matr);
    free(host_device_matr);
    cudaFree(device_mat1);
    cudaFree(device_mat2);
    cudaFree(device_matr);

    resultFile.close();     // Close File Stream

    return 0;
}
