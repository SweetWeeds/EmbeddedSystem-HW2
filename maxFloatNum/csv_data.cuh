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
public:
    CSV_Data(string fileName, bool printInfo=true);
    ~CSV_Data();
    void AddData(string Target, int numThreads, int numThreadBlks, float ExecTime);
};

#endif
