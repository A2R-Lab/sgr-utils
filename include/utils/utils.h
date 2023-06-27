/*
    Cuda utility functions
*/
#include <compiler_utils.h>
#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
CUDA_DEV T* shared_memory_proxy()
{
    // __align__(sizeof(T)) -- this will break if multiple Ts chosen
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

CUDA_DEV void sleepKernel()
{
    unsigned long long int start = clock64(); // get the start time
    unsigned long long int delay = 1000000000; // sleep for 1 second

    while (clock64() < start + delay) {
        // busy-wait
    }

    printf("Thread %d woke up after sleeping for 1 second.\n", threadIdx.x);
}

#define gpuErrchk(ans)                                                         \
    {                                                                          \
        sgrutils::gpuAssert((ans), __FILE__, __LINE__);                        \
    }

inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,
                "GPUassert: %s %s %d\n",
                cudaGetErrorString(code),
                file,
                line);
        if (abort)
            exit(code);
    }
}

template <typename T>
CUDA_HOSTDEV void printMatrixColumnMajor(T* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[j * rows + i]);
        }
        printf("\n");
    }
}

template <typename T>
CUDA_HOSTDEV void printMatrix(T* matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}