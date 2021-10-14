#ifndef CSV_DATA_H
#define CSV_DATA_H

#include <string>
#include <fstream>
#include <iostream>

using namespace std;

class CSV_Data {
private:
    ofstream resultFile;
    bool printInfo;
    float host_exec_time;
    void *host_result = NULL;
    size_t host_result_size;
    void AddData(string Target, int numThreads, int numThreadBlks, float ExecTime);
    bool CompareDeviceData(void *device_result, size_t device_result_size);
public:
    CSV_Data(string fileName, bool printInfo=true);
    ~CSV_Data();
    void AddHostData(float ExecTime, void *host_result, size_t host_result_size);
    void AddDeviceData(int numThreads, int numThreadBlks, float ExecTime, void *device_result, size_t device_result_size);
};

#endif
