
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aesctr.h"
#include <cstring>
#include <iostream>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)


__device__ void aes_fround (int secondposition, int thirdposition,int fourthposition, int& outword, int& inword,uint32_t RK,
                                    const uint32_t* __restrict__ RDFT0,
                                    const uint32_t* __restrict__ RDFT1,
                                    const uint32_t* __restrict__ RDFT2,
                                    const uint32_t* __restrict__ RDFT3
                                                                                      ){
	int ysecond = __shfl(inword, secondposition);
	int ythird = __shfl(inword,thirdposition);
	int yfourth = __shfl(inword, fourthposition);

	outword = RK^RDFT0[ ( inword   ) & 0xFF ] ^ 
		         RDFT1[ ( ysecond >>  8 ) & 0xFF ] ^
                 RDFT2[ ( ythird >> 16 ) & 0xFF ] ^   
                 RDFT3[ ( yfourth >> 24 ) & 0xFF ];    


}
__device__ void aes_finalfround (int secondposition, int thirdposition,int fourthposition, int& outword, int& inword,uint32_t RK,
                                    const unsigned char* __restrict__ RDFSb
                                                                                       ){
	int ysecond = __shfl(inword, secondposition);
	int ythird = __shfl(inword,thirdposition);
	int yfourth = __shfl(inword, fourthposition);

	outword =RK ^ ( (uint32_t) RDFSb[ ( inword       ) & 0xFF ]       ) ^
                ( (uint32_t) RDFSb[ ( ysecond >>  8 ) & 0xFF ] <<  8 ) ^
                ( (uint32_t) RDFSb[ ( ythird >> 16 ) & 0xFF ] << 16 ) ^
                ( (uint32_t) RDFSb[ ( yfourth >> 24 ) & 0xFF ] << 24 );  

                

}

__global__ void aeskernel(aes_context *ctx,
                       const unsigned char *nonce_counter,
					   unsigned char *inout,
					   size_t pitch_nc,
					   size_t pitch_io,
                                    const uint32_t* __restrict__ RDFT0,
                                    const uint32_t* __restrict__ RDFT1,
                                    const uint32_t* __restrict__ RDFT2,
                                    const uint32_t* __restrict__ RDFT3,
                                    const unsigned char* __restrict__ RDFSb
					   )
{
    int idx = threadIdx.x;
	int  xword, yword; 
	int whichoramblock = idx/16; 
	int whichcipherblock = idx/4%4; 
	int laneid = idx%32;
	int whichword = idx%4;
    uint32_t* RK_ptr = ctx->buf;

	//GET_UINT32_LE( xword, &nonce_counter[whichoramblock*pitch_nc+whichcipherblock*16+whichword*4],  whichword*4 ); 
        xword =  *((int*)  &nonce_counter[whichoramblock*pitch_nc+whichcipherblock*16+whichword*4]  );
     //   if (idx == 0)  printf("xword is 0x%x , whichword is %d\n" , xword,whichword); 
	xword ^= *(RK_ptr+whichword);
	RK_ptr += 4; 
	int secondposition = (whichword+1)%4-whichword+laneid;
	int thirdposition = (whichword+2)%4-whichword+laneid; 
	int fourthposition = (whichword+3)%4-whichword+laneid;
	for( int i = (ctx->nr >> 1) - 1; i > 0; i-- )
        {
            aes_fround( secondposition,thirdposition,fourthposition, yword, xword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);
			RK_ptr += 4; 
            aes_fround( secondposition,thirdposition,fourthposition, xword, yword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);
			RK_ptr += 4; 
        }
            aes_fround( secondposition,thirdposition,fourthposition, yword, xword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);

		RK_ptr += 4; 
            aes_finalfround( secondposition,thirdposition,fourthposition, xword, yword,*(RK_ptr+whichword),RDFSb);

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
			memcpy( &buf[64*i+j*16], aes_test_ctr_pt[0], len );
		}
	}
    memcpy( key, aes_test_ctr_key[0], 16 );
	 offset = 0;
     aes_context::aes_setkey_enc( &ctx, key);

	size_t pitch_nc, pitch_buf,pitch_sb;
	unsigned char* dnonce_counter; 
	unsigned char* dbuf ; 
	aes_context* dctx;

        uint32_t* RDFT0;
        uint32_t* RDFT1;
        uint32_t* RDFT2;
        uint32_t* RDFT3;
        unsigned char* RDFSb ; 
	CUDA_CALL(cudaMalloc(&RDFT0, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&RDFT1, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&RDFT2, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&RDFT3, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&RDFSb, sizeof(char)*256));

	//allocate device memory 
	//unsigned char* stream_block;
	CUDA_CALL(cudaMalloc(&dctx, sizeof(aes_context)));
	CUDA_CALL(cudaMallocPitch(&dnonce_counter,&pitch_nc, width, height));
	CUDA_CALL(cudaMallocPitch(&dbuf, &pitch_buf, width, height));
        
        std::cout << "nc pitch " << pitch_nc << " "; 
        std::cout << "buf pitch " << pitch_buf << std::endl; 
        std::cout << "rk is " << std::hex<< (long)(ctx.rk)<<std::endl;
        std::cout << "rk real is " << std::hex<< (long)(ctx.buf)<<std::endl;
	//copy data to device memory 
	CUDA_CALL(cudaMemcpy2D(dnonce_counter,pitch_nc,nonce_counter,width,width,height,cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy2D(dbuf,pitch_buf,buf,width,width,height,cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dctx,&ctx,sizeof(aes_context),cudaMemcpyHostToDevice));
   // cudaMallocPitch(&stream_block, &pitch_sb, sizeof(float)*width, height);

	//copy constant data to device memory 
	/* CUDA_CALL(cudaMemcpyToSymbol    (  DFT0,  FT0,   sizeof(uint32_t)*256  ));
	 CUDA_CALL(cudaMemcpyToSymbol    (  DFT1,  FT1,   sizeof(uint32_t)*256  ));
	 CUDA_CALL(cudaMemcpyToSymbol    (  DFT2,  FT2,   sizeof(uint32_t)*256  ));
	 CUDA_CALL(cudaMemcpyToSymbol    (  DFT3,  FT3,   sizeof(uint32_t)*256  ));
	 CUDA_CALL(cudaMemcpyToSymbol    (  DFSb,  FSb,   sizeof(char)*256  ));*/

	 CUDA_CALL(cudaMemcpy(  RDFT0,  FT0,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice  ));
	 CUDA_CALL(cudaMemcpy(  RDFT1,  FT1,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice  ));
	 CUDA_CALL(cudaMemcpy(  RDFT2,  FT2,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice  ));
	 CUDA_CALL(cudaMemcpy(  RDFT3,  FT3,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice ));
	 CUDA_CALL(cudaMemcpy(  RDFSb,  FSb,   sizeof(char)*256, cudaMemcpyHostToDevice  ));
	aeskernel<<<1,height*width/4>>>(dctx,dnonce_counter,dbuf,pitch_nc,pitch_buf,
                       RDFT0,RDFT1,RDFT2,RDFT3,RDFSb);
         cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy2D(buf,width,dbuf,pitch_buf,width,height,cudaMemcpyDeviceToHost));
//	aeskernel<<<1,height*width/4>>>(dctx,dnonce_counter,dbuf,pitch_nc,pitch_buf);
//    CUDA_CALL(cudaMemcpy2D(buf,width,dbuf,pitch_buf,width,height,cudaMemcpyDeviceToHost));
     std::cout<< "GPU AESCTR encryption done"<< std::endl; 
       bool encright=true ; 
	for ( int i = 0; i< height; i++){
		for (int j = 0; j<width/16; j++){
   			if( memcmp( &buf[width*i+j*16], aes_test_ctr_ct[0], len ) != 0 ){
                           encright = false; 
                           std::cout << "ORAMblock "<<i << "cipher " << j << "is incorrect" <<std::endl;	
                           std::cout << "expect 0x";
                           for (int k = 0 ; k<16;k++){
                                std::cout << std::hex<< static_cast<int>(aes_test_ctr_ct[0][k]);                              
                           }
                           std::cout<<std::endl; 
                           std::cout<< "actual 0x" ;
                           for (int k = 0 ; k<16;k++){
                                std::cout << std::hex<< static_cast<int>(buf[width*i+j*16+k]);                              
                           }
                           std::cout<<std::endl; 
			 }	
		}
	}
        if (encright) {
             std::cout<< "encryption test passed"<< std::endl; 
        } else {
             std::cout<< "encryption test failed"<< std::endl; 
        }   
	aeskernel<<<1,height*width/4>>>(dctx,dnonce_counter,dbuf,pitch_nc,pitch_buf,
                       RDFT0,RDFT1,RDFT2,RDFT3,RDFSb);
         cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy2D(buf,width,dbuf,pitch_buf,width,height,cudaMemcpyDeviceToHost));
     std::cout<< "GPU AESCTR decryption done"<< std::endl; 
       bool decright=true ; 
	for ( int i = 0; i< height; i++){
		for (int j = 0; j<width/16; j++){
   			if( memcmp( &buf[width*i+j*16], aes_test_ctr_pt[0], len ) != 0 ){
                           decright = false; 
                           std::cout << "ORAMblock "<<i << "cipher " << j << "is incorrect" <<std::endl;	
                           std::cout << "expect 0x";
                           for (int k = 0 ; k<16;k++){
                                std::cout << std::hex<< static_cast<int>(aes_test_ctr_pt[0][k]);                              
                           }
                           std::cout<<std::endl; 
                           std::cout<< "actual 0x" ;
                           for (int k = 0 ; k<16;k++){
                                std::cout << std::hex<< static_cast<int>(buf[width*i+j*16+k]);                              
                           }
                           std::cout<<std::endl; 
			 }	
		}
	}
        if (decright) {
             std::cout<< "decryption test passed"<< std::endl; 
        } else {
             std::cout<< "decryption test failed"<< std::endl; 
        }   
 
       std::cout<< "test done" << std::endl; 
	delete[] buf; 
	delete[] nonce_counter;
        cudaFree (dbuf)	;
        cudaFree(dnonce_counter);
        cudaFree(dctx);
        cudaFree(RDFT0);
        cudaFree(RDFT1);
        cudaFree(RDFT2);
        cudaFree(RDFT3);
        cudaFree(RDFSb);
    CUDA_CALL( cudaDeviceReset());
    return 0;
}
