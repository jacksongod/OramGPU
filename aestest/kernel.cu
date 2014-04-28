
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aesctr.h"
#include <cstring>
#include <iostream>
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

	outword = RK^DFT0[ ( inword   ) & 0xFF ] ^ 
		         DFT1[ ( ysecond >>  8 ) & 0xFF ] ^
                 DFT2[ ( ythird >> 16 ) & 0xFF ] ^   
                 DFT3[ ( yfourth >> 24 ) & 0xFF ];    


}
__device__ void aes_finalfround (int idx, int whichword,int laneid, int& outword, int& inword,uint32_t RK){
	int secondposition = (whichword+1)%4-whichword+laneid;
	int thirdposition = (whichword+2)%4-whichword+laneid; 
	int fourthposition = (whichword+3)%4-whichword+laneid;
	int ysecond = __shfl(inword, secondposition);
	int ythird = __shfl(inword,thirdposition);
	int yfourth = __shfl(inword, fourthposition);

	outword =RK ^ ( (uint32_t) DFSb[ ( inword       ) & 0xFF ]       ) ^
                ( (uint32_t) DFSb[ ( ysecond >>  8 ) & 0xFF ] <<  8 ) ^
                ( (uint32_t) DFSb[ ( ythird >> 16 ) & 0xFF ] << 16 ) ^
                ( (uint32_t) DFSb[ ( yfourth >> 24 ) & 0xFF ] << 24 );  

                

}

__global__ void aeskernel(aes_context *ctx,
                     //  size_t length,
                     //  size_t *nc_off,
                    //   unsigned char nonce_counter[16],
                    //   unsigned char stream_block[16],
                       const unsigned char *nonce_counter,
                      // unsigned char **stream_block,
					  // unsigned char **input,
					   unsigned char *inout,
					   size_t pitch_nc,
					   size_t pitch_io
					   )
{
    int idx = threadIdx.x;
	int xword, yword; 
	int whichoramblock = idx/16; 
	int whichcipherblock = idx/4%4; 
	int laneid = idx%32;
	int whichword = idx%4;
    uint32_t* RK_ptr = ctx->rk;

	GET_UINT32_LE( xword, &nonce_counter[whichoramblock*pitch_nc+whichcipherblock*16+whichword*4],  whichword*4 ); 
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

//		PUT_UINT32_LE( xword, &stream_block[whichoramblock][whichcipherblock*16+whichword*4],  whichword*4);
		*((uint32_t*)&inout[whichoramblock*pitch_io+whichcipherblock*16+whichword*4]) = 
			*((uint32_t*)&inout[whichoramblock*pitch_io+whichcipherblock*16+whichword*4])^
			xword; 
			//*((uint32_t*)&stream_block[whichoramblock][whichcipherblock*16+whichword*4]);

}

int main()
{
    // int i, j, u, v;
    unsigned char key[16];
    
    unsigned char iv[16];


    size_t offset;

    int len;
	int height = 24;     //num of oram blocks; 
	int width = 64;      //num of bytes in one oram block
    unsigned char* nonce_counter = new unsigned char[height*width];
  //  unsigned char* stream_block= new unsigned char[height*width];
	unsigned char* buf= new unsigned char[height*width];
    aes_context ctx;
	 len = 16;
	for ( int i = 0; i< height; i++){

		for (int j = 0; j<width/16; j++){
			memcpy( &nonce_counter[64*i+j*16], aes_test_ctr_nonce_counter[0], 16 );
			memcpy( &buf[64*i+j*16], aes_test_ctr_ct[0], len );
		}
	}
    memcpy( key, aes_test_ctr_key[0], 16 );
	 offset = 0;
     aes_context::aes_setkey_enc( &ctx, key);

	size_t pitch_nc, pitch_buf,pitch_sb;
	unsigned char* dnonce_counter; 
	unsigned char* dbuf ; 
	aes_context* dctx;

	//allocate device memory 
	//unsigned char* stream_block;
	CUDA_CALL(cudaMalloc(&dctx, sizeof(aes_context)));
	CUDA_CALL(cudaMallocPitch(&dnonce_counter,&pitch_nc, width, height));
	CUDA_CALL(cudaMallocPitch(&dbuf, &pitch_buf, width, height));

	//copy data to device memory 
	CUDA_CALL(cudaMemcpy2D(dnonce_counter,pitch_nc,buf,width,width,height,cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy2D(dbuf,pitch_buf,buf,width,width,height,cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dctx,&ctx,sizeof(aes_context),cudaMemcpyHostToDevice));
   // cudaMallocPitch(&stream_block, &pitch_sb, sizeof(float)*width, height);

	//copy constant data to device memory 
	 cudaMemcpyToSymbol    (  DFT0,  FT0,   sizeof(uint32_t)*256  );
	 cudaMemcpyToSymbol    (  DFT1,  FT1,   sizeof(uint32_t)*256  );
	 cudaMemcpyToSymbol    (  DFT2,  FT2,   sizeof(uint32_t)*256  );
	 cudaMemcpyToSymbol    (  DFT3,  FT3,   sizeof(uint32_t)*256  );
	 cudaMemcpyToSymbol    (  DFSb,  FSb,   sizeof(char)*256  );
	aeskernel<<<1,height*width>>>(dctx,dnonce_counter,dbuf,pitch_nc,pitch_buf);
    CUDA_CALL(cudaMemcpy2D(buf,width,dbuf,pitch_buf,width,height,cudaMemcpyDeviceToHost));
    // Add vectors in parallel.
	
 
    CUDA_CALL( cudaDeviceReset());

	delete[] buf; 
	delete[] nonce_counter;
	
    return 0;
}
