
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aesctr.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);




__global__ void aeskernel(aes_context *ctx,
                     //  size_t length,
                     //  size_t *nc_off,
                    //   unsigned char nonce_counter[16],
                    //   unsigned char stream_block[16],
                       const unsigned char *input,
                       unsigned char *output )
{
    int i = threadIdx.x;

}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
