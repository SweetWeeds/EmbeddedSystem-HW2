#include "csv_data.cuh"

CSV_Data::CSV_Data(string fileName, bool printInfo) {
    (this->resultFile).open(fileName, ios::out);
    (this->resultFile) << "Target,#Threads,#ThreadBlks,ExecTime\n";
    this->printInfo = printInfo;
}


CSV_Data::~CSV_Data() {
    resultFile.close();
}


void CSV_Data::AddData(string Target, int numThreads, int numThreadBlks, float ExecTime) {
    if (printInfo) {
        printf("[INFO] Target: %10s, numThreads: %5d, numThreadBlks: %5d, ExecTime: %.5f\n",
                Target.c_str(), numThreads, numThreadBlks, ExecTime);
    }
    (this->resultFile) << Target << "," << numThreads << "," << numThreadBlks << "," << ExecTime << "\n";
}


bool CSV_Data::CompareDeviceData(void *device_result, size_t device_result_size) {
    if (this->host_result == NULL) {
        printf("[WARN] Host result data is empty.\n");
        return false;
    }
    if (this->host_result_size == device_result_size) {
        if (memcmp(device_result, this->host_result, device_result_size) == 0) {
            return true;
        } else {
            printf("[WARN] Data is not matching with host's and device's.\n");
            return false;
        }
    } else {
        printf("[WARN] Data size is not matching with host's and device's.\n"
               "        Device: %lu, Host: %lu\n", device_result_size, this->host_result_size);
        return false;
    }
}


void CSV_Data::AddHostData(float ExecTime, void *host_result, size_t host_result_size) {
    this->host_exec_time = ExecTime;
    if (this->host_result) {
        free(this->host_result);
    }
    this->host_result = (void *)malloc(host_result_size);
    this->host_result_size = host_result_size;
    memcpy(this->host_result, host_result, host_result_size);
    this->AddData("host", 1, 1, ExecTime);
}


void CSV_Data::AddDeviceData(int numThreads, int numThreadBlks, float ExecTime, void *device_result, size_t device_result_size) {
    this->CompareDeviceData(device_result, device_result_size);
    this->AddData("device", numThreads, numThreadBlks, ExecTime);
}
