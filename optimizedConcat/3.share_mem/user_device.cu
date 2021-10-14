#include "user_device.h"

__global__
void device_Concatenate(int *device_mat1, int *device_mat2, int *device_matr,
                        int numOps, int numElements, int mat1_col, int mat2_col, int matr_col) {
    __shared__ int subMatr[1024];
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int row_index, col_index;
    int offset = gridDim.x * blockDim.x;

    for (int i = 0; i < numOps; i++) {
        row_index = index / matr_col;
        col_index = index % matr_col;
        if (index < numElements) {
            subMatr[threadIdx.x] = (col_index < mat1_col) ?
                device_mat1[row_index * mat1_col + col_index] :
                device_mat2[row_index * mat2_col + (col_index - mat1_col)];
            //__syncthreads();
            device_matr[index] = subMatr[threadIdx.x];
            index += offset;
        }
    }
}
