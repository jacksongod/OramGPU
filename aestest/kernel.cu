
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aesctr.h"
#include <cstring>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);


__device__ void aes_fround (int idx, int whichword,int laneid, int& outword, int& inword,uint32_t RK){
	int secondposition = (whichword+1)%4-whichword+laneid;
	int thirdposition = (whichword+2)%4-whichword+laneid; 
	int fourthposition = (whichword+3)%4-whichword+laneid;
	int ysecond = __shfl(inword, secondposition);
	int ythird = __shfl(inword,thirdposition);
	int yfourth = __shfl(inword, fourthposition);

	outword = RK^FT0[ ( inword   ) & 0xFF ] ^ 
		         FT1[ ( ysecond >>  8 ) & 0xFF ] ^
                 FT2[ ( ythird >> 16 ) & 0xFF ] ^   
                 FT3[ ( yfourth >> 24 ) & 0xFF ];    


}
__device__ void aes_finalfround (int idx, int whichword,int laneid, int& outword, int& inword,uint32_t RK){
	int secondposition = (whichword+1)%4-whichword+laneid;
	int thirdposition = (whichword+2)%4-whichword+laneid; 
	int fourthposition = (whichword+3)%4-whichword+laneid;
	int ysecond = __shfl(inword, secondposition);
	int ythird = __shfl(inword,thirdposition);
	int yfourth = __shfl(inword, fourthposition);

	outword =RK ^ ( (uint32_t) FSb[ ( inword       ) & 0xFF ]       ) ^
                ( (uint32_t) FSb[ ( ysecond >>  8 ) & 0xFF ] <<  8 ) ^
                ( (uint32_t) FSb[ ( ythird >> 16 ) & 0xFF ] << 16 ) ^
                ( (uint32_t) FSb[ ( yfourth >> 24 ) & 0xFF ] << 24 );  

                

}

__global__ void aeskernel(aes_context *ctx,
                     //  size_t length,
                     //  size_t *nc_off,
                    //   unsigned char nonce_counter[16],
                    //   unsigned char stream_block[16],
                       const unsigned char **nonce_counter,
                       unsigned char **stream_block,
					   unsigned char **input,
					   unsigned char **output
					   )
{
    int idx = threadIdx.x;
	int xword, yword; 
	int whichoramblock = idx/16; 
	int whichcipherblock = idx/4%4; 
	int laneid = idx%32;
	int whichword = idx%4;
    uint32_t* RK_ptr = ctx->rk;

	GET_UINT32_LE( xword, &nonce_counter[whichoramblock][whichcipherblock*16+whichword*4],  whichword*4 ); 
	xword ^= *(RK_ptr+whichword);
	RK_ptr += 4; 
	for( int i = (ctx->nr >> 1) - 1; i > 0; i-- )
        {
            aes_fround( idx, whichword, laneid, yword, xword,*(RK_ptr+whichword));
			RK_ptr += 4; 
            aes_fround( idx, whichword, laneid, xword, yword,*(RK_ptr+whichword));
			RK_ptr += 4; 
        }

        aes_fround( idx, whichword, laneid, yword, xword,*(RK_ptr+whichword) );
		RK_ptr += 4; 
		aes_finalfround ( idx, whichword, laneid, xword, yword,*(RK_ptr+whichword) );

		PUT_UINT32_LE( xword, &stream_block[whichoramblock][whichcipherblock*16+whichword*4],  whichword*4);
		*((uint32_t*)&output[whichoramblock][whichcipherblock*16+whichword*4]) = 
			*((uint32_t*)&input[whichoramblock][whichcipherblock*16+whichword*4])^
			*((uint32_t*)&stream_block[whichoramblock][whichcipherblock*16+whichword*4]);

}

int main()
{
     int i, j, u, v;
    unsigned char key[16];
    unsigned char buf[64];
    unsigned char iv[16];

    size_t offset;

    int len;
    unsigned char nonce_counter[16];
    unsigned char stream_block[16];
    aes_context ctx;

	memcpy( nonce_counter, aes_test_ctr_nonce_counter[0], 16 );
    memcpy( key, aes_test_ctr_key[0], 16 );
	 offset = 0;

	 
     aes_context::aes_setkey_enc( &ctx, key);
	 len = aes_test_ctr_len[0];
     memcpy( buf, aes_test_ctr_ct[0], len );
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
