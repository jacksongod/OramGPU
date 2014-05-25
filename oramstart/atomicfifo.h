#include<stdio.h>

#define STASHSIZE 256

__device__ __forceinline__ void getlock(uint32_t* mutex ){
   while (atomicCAS(mutex,0,1) !=0);
}
__device__ __forceinline__ bool getlockattempt(uint32_t* mutex){
   return (atomicCAS(mutex,0,1) ==0);
}
__device__ __forceinline__ void releaselock(uint32_t* mutex ){
   atomicExch(mutex,0);
}



__device__ __forceinline__ uint32_t fifopop(uint32_t* fifo, int32_t* front, int32_t* end) {
  if (*front == *end){
	printf("attempt to pop from an empty fifo\n");
  } 
  uint32_t temp = fifo[*end];
  *end= (*end+1) % STASHSIZE; 
  return temp;   
}

__device__ __forceinline__ void fifopush(uint32_t* fifo, int32_t* front, int32_t* end,uint32_t val) {
  if (*front == ((*end+1)%STASHSIZE)){
	printf("attempt to push to a full fifo\n");
  } 
  fifo[*front] = val; 
  *front = (*front+1)%STASHSIZE; 

}
