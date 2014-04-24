
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aesctr.h"
#include <cstring>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__device__ void aes_fround (int idx, int whichbyte, int outword, int inword){


}


__global__ void aeskernel(aes_context *ctx,
                     //  size_t length,
                     //  size_t *nc_off,
                    //   unsigned char nonce_counter[16],
                    //   unsigned char stream_block[16],
                       const unsigned char *input,
                       unsigned char *output )
{
    int idx = threadIdx.x;
	int xword, yword; 
	int whichoramblock = idx/16; 
	int whichcipherblock = idx/4%4; 
	int laneid = idx%32;
	int whichbyte = idx%4;
    uint32_t* RK = ctx->rk;

    GET_UINT32_LE( xword, input,  whichbyte*4 ); xword ^= *(RK+whichbyte);
	RK += 4; 
	for( int i = (ctx->nr >> 1) - 1; i > 0; i-- )
        {
            AES_FROUND( Y0, Y1, Y2, Y3, X0, X1, X2, X3 );
            AES_FROUND( X0, X1, X2, X3, Y0, Y1, Y2, Y3 );
        }

        AES_FROUND( Y0, Y1, Y2, Y3, X0, X1, X2, X3 );

}

int main()
{
     int i, j, u, v;
    unsigned char key[32];
    unsigned char buf[64];
    unsigned char iv[16];

    size_t offset;

    int len;
    unsigned char nonce_counter[16];
    unsigned char stream_block[16];
    aes_context ctx;

    memset( key, 0, 32 );
	memcpy( nonce_counter, aes_test_ctr_nonce_counter[0], 16 );
    memcpy( key, aes_test_ctr_key[0], 16 );
	 offset = 0;

	 
     aes_context::aes_setkey_enc( &ctx, key);
	 len = aes_test_ctr_len[u];
     memcpy( buf, aes_test_ctr_ct[u], len );
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
