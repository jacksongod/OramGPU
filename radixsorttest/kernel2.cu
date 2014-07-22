#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "aesctr.h"
#include <cstring>
#include <iostream>
#include <cub/cub.cuh>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)
#define CUDA_ARCH 350
                
__global__ void radixsortkernel(uint32_t *in_buf,
				uint32_t *keyin_buf, 
                                   uint32_t* out_buf,
                                   uint32_t* keyout_buf
					   )
{
	typedef cub::BlockLoad<uint32_t*, 256, 1, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
	typedef cub::BlockStore<uint32_t*, 256, 1, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
	typedef cub::BlockRadixSort<uint32_t, 256, 1,uint32_t/*, 4,true,cub::BLOCK_SCAN_RAKING*/  > BlockRadixSort;
	__shared__ union {
        	typename BlockLoad::TempStorage       load; 
       		typename BlockStore::TempStorage      store; 
        	typename BlockRadixSort::TempStorage  sort;
   	} temp_storage; 
	__shared__ union {
        	typename BlockLoad::TempStorage       load; 
       		typename BlockStore::TempStorage      store; 
   	} key_storage; 
	//if (threadIdx.x<64){
	uint32_t v_data[1];
	uint32_t k_data[1]; 
	BlockLoad(temp_storage.load).Load( in_buf,v_data);
	BlockLoad(key_storage.load).Load( keyin_buf,k_data);
	__syncthreads(); 
	BlockRadixSort(temp_storage.sort).Sort(k_data,v_data);
	__syncthreads(); 
	BlockStore(temp_storage.store).Store(out_buf, v_data );
        keyout_buf[threadIdx.x] = k_data[0];	
	//BlockStore(key_storage.store).Store(keyout_buf, k_data );
	//}
	
}

int main()
{
    // int i, j, u, v;
       const int size = 256; 
	uint32_t hin_array[size] ;
	uint32_t hkey_array[size] ;
	uint32_t hout_array[size] ; 
	uint32_t hkeyout_array[size] ; 
	for (int i = 0 ; i<size; i++ ){
		hin_array[i] = i;
		hkey_array[i] =(i%2==0)? 2:1;
	}
//        hkey_array[size-1] = 1;
	std::cout << "input value : "<< std::endl; 
	for (int i = 0; i<size ; i++){
		std::cout<< hin_array[i] << " ";
	}
	std::cout<< std::endl; 
	std::cout << "input key : "<< std::endl; 
	for (int i = 0; i<size ; i++){
		std::cout<< hkey_array[i] << " ";
	}
	std::cout<< std::endl; 
        uint32_t* in_arr;
        uint32_t* key_arr;
        uint32_t* out_arr;
        uint32_t* keyout_arr;
	CUDA_CALL(cudaMalloc(&in_arr, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&key_arr, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&out_arr, sizeof(uint32_t)*256));
	CUDA_CALL(cudaMalloc(&keyout_arr, sizeof(uint32_t)*256));

	//allocate device memory 
	//unsigned char* stream_block;
        
	//copy data to device memory 
	CUDA_CALL(cudaMemcpy(in_arr,hin_array,sizeof(uint32_t)*256,cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(key_arr,hkey_array,sizeof(uint32_t)*256,cudaMemcpyHostToDevice));

	//copy constant data to device memory 

	radixsortkernel<<<1,size>>>(in_arr,key_arr,
                       out_arr,keyout_arr);
         cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(hout_array,out_arr,sizeof(uint32_t)*256,cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(hkeyout_array,keyout_arr,sizeof(uint32_t)*256,cudaMemcpyDeviceToHost));
//	aeskernel<<<1,height*width/4>>>(dctx,dnonce_counter,dbuf,pitch_nc,pitch_buf);
//    CUDA_CALL(cudaMemcpy2D(buf,width,dbuf,pitch_buf,width,height,cudaMemcpyDeviceToHost));
     std::cout<< "GPU Radix sort done"<< std::endl; 
       bool encright=true ; 
	std::cout << "output value : "<< std::endl; 
	for (int i = 0; i<size ; i++){
		std::cout<< hout_array[i] << " ";
	}
	std::cout<< std::endl; 
	std::cout << "output key : "<< std::endl; 
	int old = 0;
	for (int i = 0; i<size ; i++){
		if (old> hkeyout_array[i]){
			encright = false; 
		}
		old = hkeyout_array[i];
		std::cout<< hkeyout_array[i] << " ";
	}
	std::cout<< std::endl; 
        if (encright) {
             std::cout<< "Sort test passed"<< std::endl; 
        } else {
             std::cout<< "Sort test failed"<< std::endl; 
        }   
 
       std::cout<< "test done" << std::endl; 
        cudaFree (in_arr)	;
        cudaFree(out_arr);
    CUDA_CALL( cudaDeviceReset());
    return 0;
}
