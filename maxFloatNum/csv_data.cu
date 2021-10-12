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
