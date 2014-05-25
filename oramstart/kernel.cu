
// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "bucket.h"
#include <curand_kernel.h>
#include <cuda.h>
#include "atomicfifo.h"

#define CUDATHREADNUMLOG 9
#define CUDATHREADNUM (1<<CUDATHREADNUMLOG)


#define BLOCKNUMLOG 12
#define MAPSIZEPERTHREAD 8
#define BLOCKSIZE 64
#define LEAFNUMLOG  11
#define TREESIZE (1<<LEAFNUMLOG)*2//-1
#define CUDABLOCKNUM 1
#define BLOCKPERBUCKET 2

#define STASHSIZE 256

#define ACCESSNUM 1000
/**
 * Main
 */
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

typedef TBucket<BLOCKPERBUCKET> OramB; 
typedef TDBucket<BLOCKPERBUCKET,BLOCKSIZE> OramD;



__device__ __forceinline__ int calcindex(int level, uint16_t path){
    return (path>>(LEAFNUMLOG-level))+ (1<<level) -1 ;

}

__global__ void oramshare(uint16_t* position_table, uint32_t* access_script,uint16_t* checktable, OramB* oramtree, TDBlock<BLOCKSIZE>* checktable2,curandState *randstate,OramD* datatree){
	int tid =  threadIdx.x;
        curandState localrandState = randstate[tid];
        __shared__ OramB metatree[TREESIZE];              //4K*sizeof(OramB) = 16KB
       // __shared__ OramB treepath[LEAFNUMLOG+1];          //12*sizeof(OramB) = 48B
        __shared__ uint32_t treepathlock[(LEAFNUMLOG+1)*2];    //96B
        __shared__ uint32_t streepathlock[STASHSIZE];    //1kB
        __shared__ uint16_t stash [STASHSIZE];            //256B
        __shared__ uint32_t expectedblockindex; 
   //     __shared__ uint32_t stashlock[STASHSIZE];          //1kB
        __shared__ uint32_t camfifo[STASHSIZE]; 
        __shared__ int32_t front,end;
        __shared__ uint32_t  mutex; 
	__shared__ uint16_t localtable[1<<(BLOCKNUMLOG)];   //8KB  when blocknumlog = 12
        __shared__  uint16_t newposition; 
        __shared__ uint32_t stashcount;
		__shared__ uint32_t maxstashcount;
        __shared__ uint32_t pathcount;
        __shared__ uint32_t stashaccessloc[(LEAFNUMLOG+1)*BLOCKPERBUCKET]; //2*12*4B = 96B
        __shared__ uint32_t writebackloc[(LEAFNUMLOG+1)*BLOCKPERBUCKET];   //2*12*4B = 96B
        __shared__ uint32_t  datastash[STASHSIZE*(BLOCKSIZE/4)];          //4B * 256*16 = 16KB
		//__shared__ TDBlock<BLOCKSIZE> garbage_collector; 
		//__shared__ uint32_t blockinstash;  
       
    //copy position table from global memory to shared memory
     //  localtable[tid*MAPSIZEPERTHREAD] = position_table[tid*MAPSIZEPERTHREAD];
     //  localtable[tid*MAPSIZEPERTHREAD+1] = position_table[tid*MAPSIZEPERTHREAD+1];
     //  localtable[tid*MAPSIZEPERTHREAD+2] = position_table[tid*MAPSIZEPERTHREAD+2];
     //  localtable[tid*MAPSIZEPERTHREAD+3] = position_table[tid*MAPSIZEPERTHREAD+3];
     //localtable[tid*2] = (position_table[tid*MAPSIZEPERTHREAD+1]<<16) | position_table[tid*MAPSIZEPERTHREAD];
     //localtable[tid*2+1] = (position_table[tid*MAPSIZEPERTHREAD+3]<<16) | position_table[tid*MAPSIZEPERTHREAD+2];
 
    memcpy(&localtable[tid*MAPSIZEPERTHREAD],&position_table[tid*MAPSIZEPERTHREAD],sizeof(uint16_t)*MAPSIZEPERTHREAD);
    // copy metadata tree from global memory to shared memory
    memcpy(&metatree[tid*MAPSIZEPERTHREAD], &oramtree[tid*MAPSIZEPERTHREAD],sizeof(OramB)*MAPSIZEPERTHREAD);
   if (tid <256) {
	// stashlock[tid] = 0;
       camfifo[tid] = tid; 
       stash[tid] = 0xffff; 
   }
  // if (tid< 24) treepathlock[tid] = 0;
  // if (tid<12) streepathlock[tid] = 0;
   //if (tid ==256) pathcount = 24;
   if (tid ==511) {
	stashcount = STASHSIZE;
         end = mutex = 0; 
        front = STASHSIZE -1; 
   }
   if(tid == 384) maxstashcount = 256;
   
  //  if (tid< (LEAFNUMLOG+1)){
  //  memset(&treepathlock[tid],0x0,4);
    
  //  }//else if (tid==1023){
  //   stashcount =0;
  //   pathcount = 0;
  //  }
   __syncthreads();
    
                       //checktable[tid] = 1;
    uint32_t pathid; 
    //uint32_t pathidtemp;
    //uint32_t accessid; 
    //bool r_foundposition=false;
    //bool w_foundposition=false;; 
   // int startindex= tid/2 ; //tid *8
    uint16_t blockid;
	for (int i = 0 ; i<ACCESSNUM ; i++ ){
               blockid = access_script[i];
               pathid = localtable[blockid];
         //      if ((blockid>>3) == (tid-512)){
                   newposition =(curand(&localrandState))& 0x7ff; 
          //     }
            //    accessid = access_script[i];
	    //	pathidtemp = localtable[accessid/2] ;
            //    pathid =   0xffff&(pathidtemp  >>((accessid&0x1)<<4)); 
  //       int myblockid ;
       // if(tid <STASHSIZE)  {
	//		streepathlock[tid] = 0;}
		//	 myblockid = stash[tid] ;
		//	if(blockid == myblockid &&stashlock[tid] !=0){
		//		blockinstash = 1; 
		//		checktable[i] = blockid;
		//	} else{
		//		blockinstash =0; 
		//	}
		//} 
  //      __syncthreads();

		//if (!blockinstash) {
         if (tid< 384 && tid%16==0){  //copy entire path to local registers (12 levels, 24 blocks)
               int stid = tid /16;
               treepathlock[stid] = 0;
                //pathcount = 24; 
            //printf("rand : %d is %d\n",tid,  (unsigned)(curand(&localrandState))%(1<<LEAFNUMLOG));
           int treeindex = calcindex(stid/2, pathid);
           //printf("id: %d, index %d\t",tid,treeindex );
           uint16_t id = metatree[treeindex].id[stid%2];
		   metatree[treeindex].id[stid%2] = 0;
           if((id>>15) == 1){   // if data is valid 
            // printf("id: %d valid data \t ", tid); 
              /*while(true){
                 if (!atomicCAS(&stashlock[startindex],0,1 )){
            //         printf("id: %d, foundposition\t",tid);
                    stashaccessloc[stid] = startindex;
                    stash[startindex] = id &0xfff; 
                    
                 startindex = (startindex+1)%STASHSIZE; 
                  //  atomicSub(&stashcount,1);
              //       printf("id: %d, data id %d\n",tid,stash[startindex%STASHSIZE]);
              //      checktable[i*24+tid] = stash[startindex] ;
              //      checktable2[i*24+tid] = pathid;
              //      printf("after id %d, checktable data %d \n", tid, checktable[i*24+tid]);
                    break; 
                 }
                 startindex = (startindex+1)%STASHSIZE; 
	       } */
            //   printf("out\n");
               bool needlock = true; 
               int positionindex ; 
                       while(needlock) { 
                       	if (getlockattempt(&mutex)) {
                                positionindex = fifopop(camfifo, &front,&end);
                       		releaselock(&mutex);
                                needlock = false; 
                        }

                       }
              stashaccessloc[stid] = positionindex;
              stash[positionindex] = id & 0xfff;
		 }else{
		   stashaccessloc[stid] = 999;
         	   }
		   
                     
         }  
         __syncthreads();
        // if (tid==0 ) atomicMin(&maxstashcount, stashcount);
	//	 if(i==80){

	//		 int myball = 1000;
	//	 }
		
         if (tid < STASHSIZE){
             if (stash[tid]!=0xffff){
		   int myblockid = stash[tid];
		   if(myblockid == blockid ){
                      localtable[myblockid] = newposition;
                      checktable[i] = blockid;
                      expectedblockindex = tid;
		   }
                      
                   int sortkey = localtable[myblockid] ^ pathid;  
                   int level = __clz((sortkey<<21)|0x00100000); 
                   int treeindex = calcindex(level,pathid);
                while(true){
				   int blockloc = (level<<1);
                   if(!atomicCAS(&treepathlock[level<<1],0,1)){
                       writebackloc[blockloc] = tid; 
                       metatree[treeindex].id[0] = 0x8000|myblockid;
                       bool needlock = true; 
                       while(needlock) { 
                       	if (getlockattempt(&mutex)) {
                       		fifopush(camfifo,&front,&end,tid);
                       		releaselock(&mutex);
                                needlock = false; 
                        }

                       }
                      stash[tid] = 0xffff;
                      // stashlock[tid] = 0;
                //       atomicAdd(&stashcount,1);
                    //   atomicSub(&pathcount,1);
                       break; 
                   } else if (!atomicCAS(&treepathlock[(level<<1)+1],0,1)){
                       writebackloc[blockloc+1] = tid; 
                       metatree[treeindex].id[1] = 0x8000|myblockid;
                       bool needlock = true; 
                       while(needlock) { 
                       	if (getlockattempt(&mutex)) {
                       		fifopush(camfifo,&front,&end,tid);
                       		releaselock(&mutex);
                                needlock = false; 
                        }

                       }
                      stash[tid] = 0xffff;
                      // stashlock[tid] = 0;
              //         atomicAdd(&stashcount,1);
                     //  atomicSub(&pathcount,1);
                       break;
                   } 
                   level--; 
                   if (level<0) break;   
                   treeindex = (treeindex-1)>>1;
                    
                   
                }		       

             }


         } 
         //if (tid <256)  stashlock[tid] = 0;      
      /*   if (tid < STASHSIZE*BLOCKPERBUCKET){
              bool secondblock = (tid>=STASHSIZE);
              int stid = tid-secondblock*STASHSIZE; 
             if (stashlock[stid]!=0){
		   int myblockid = stash[stid] ;
		   if(myblockid == blockid ){
                      localtable[myblockid] = newposition;
					   checktable[i] = blockid;
                      expectedblockindex = stid;
                      //stashaccessloc = stid;
		   }
                      
                   int sortkey = localtable[myblockid] ^ pathid;  
                   int level = __clz((sortkey<<21)|0x00100000); 
                   int treeindex = calcindex(level,pathid);
                while(true){
                 //  if (pathcount<=0) break;
                   int blockloc = (level<<1) +secondblock;
                   if(!atomicCAS(&treepathlock[blockloc],0,1)){
                       if (atomicCAS(&streepathlock[stid],0,1)) {
                          treepathlock[blockloc] = 0;
                          break;
                       } 
                       writebackloc[blockloc] = stid; 
                       metatree[treeindex].id[secondblock] = 0x8000|myblockid;
                       stashlock[stid] = 0;
                       atomicAdd(&stashcount,1);
                   //    atomicSub(&pathcount,1);
                       break; 
                   } 
                   level--; 
                   if (level<0) break;   
                   treeindex = (treeindex-1)/2;
                    
                   
                }		       

			 } 


         }*/
		// else if(tid<896&&tid>511) {          //other threads bring in the data from tree to data stash. 
           if (tid< 384){
              //int stid = tid - STASHSIZE*BLOCKPERBUCKET; 
              int bucketid= tid/16;
             int treeindex = calcindex(bucketid/2, pathid);
              int whichdata = tid%16;
              int whichblock = bucketid%2;
			  volatile int gabarge;
			    if(stashaccessloc[bucketid] !=999){           //999 means this block in the tree path is empty
              datastash[(stashaccessloc[bucketid]%STASHSIZE)*16+whichdata] = datatree[treeindex].block[whichblock].data[whichdata]; 
				}else{                             // if block in the tree path is empty , read the block still, but write to a garbage(don't care) position 
					gabarge =  datatree[treeindex].block[whichblock].data[whichdata];
				}

         }  
         __syncthreads();

         // writeback data back from stash to tree 
         if (tid < 384){
           int bucketid = tid/16;
             int treeindex = calcindex(bucketid/2, pathid);
              int whichdata = tid%16;
              int whichblock = bucketid%2;
			
             datatree[treeindex].block[whichblock].data[whichdata] = datastash[(writebackloc[bucketid]%STASHSIZE)*16+whichdata];       
             //if (writebackloc[bucketid] >=STASHSIZE){
             //   printf("invalllllllll\n %d access, %d thread", i,tid );
            // }

         }else if (tid< 400){
             int stid = tid-384;
	     checktable2[i].data[stid] = datastash[expectedblockindex*16+stid];   

         }  
	//	 __syncthreads();
	
     
	}

    

    //if (tid == 0)    printf("max stash size %ud\n",256-maxstashcount);
}
__global__ void setup_kernel(curandState *state)
{
int id = threadIdx.x;
/* Each thread gets same seed, a different sequence number,
no offset */
curand_init(1234, id, 0, &state[id]);
}


int main(int argc, char** argv)
{
    // Initialize command line
	 cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
//	 cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
      
    printf("start\n");
   curandState *devStates;
   CUDA_CALL(cudaMalloc((void **)&devStates, 1024 * sizeof(curandState)));
   setup_kernel<<<CUDABLOCKNUM,CUDATHREADNUM>>>(devStates);
    uint16_t* p_table = new uint16_t[1<<BLOCKNUMLOG];
    uint16_t* check_table = new uint16_t[ACCESSNUM];
    TDBlock<BLOCKSIZE>* check_table2 = new TDBlock<BLOCKSIZE>[ACCESSNUM];
	TDBlock<BLOCKSIZE>* resultlist = new TDBlock<BLOCKSIZE>[1<<BLOCKNUMLOG];
    uint32_t* access_script = new uint32_t[ACCESSNUM];
    //uint32_t* orampath = new uint32_t[1<<LEAFNUMLOG];
    OramB* oramtree = new OramB[TREESIZE];
    OramD* doramtree = new OramD[TREESIZE];

    printf("sizeof OramD %d\n", sizeof(OramD));
    for (int i = 0; i< (1<<(BLOCKNUMLOG)); i++){
        p_table[i] = 0xdead;
    }
    printf("finished initialize raw p_table\n");
    int startpoint =  (1<<LEAFNUMLOG) -1 ; 
    for (int i =0; i< startpoint;i++){
	oramtree[i].initzero();

    }
    for (int i = startpoint; i< (TREESIZE-1); i++){
            int temp = i- startpoint;
        doramtree[i].init(); 
        oramtree[i].init(temp*BLOCKPERBUCKET); 
        p_table[BLOCKPERBUCKET*temp] = temp;
        p_table[BLOCKPERBUCKET*temp+1] = temp;
        resultlist[BLOCKPERBUCKET*temp] =doramtree[i].block[0];
		resultlist[BLOCKPERBUCKET*temp+1] =doramtree[i].block[1];

        //oramtree[i].id[0] =( 0x8000 |rand()%(1<<BLOCKNUMLOG)); 
        //p_table[oramtree[i].id[0]%(1<<BLOCKNUMLOG)] = temp; 
        //oramtree[i].id[1] =( 0x8000| rand()%(1<<BLOCKNUMLOG)); 
        //p_table[oramtree[i].id[1]%(1<<BLOCKNUMLOG)] = temp; 
    }
    printf ("finished initializa p_table \n");
    printf ("Accessing %d blocks \n", ACCESSNUM);

    for (int i = 0; i<(1<BLOCKNUMLOG); i++){
	if (p_table[i] != i/BLOCKPERBUCKET)  printf("p_table is wrong\n");

      printf("finish checking position table\n");
    }
   
    for (int i = 0; i<(ACCESSNUM); i++){
        access_script[i] = rand()%(1<<BLOCKNUMLOG);
          if (p_table[access_script[i]] == 0xdead) printf("p_table has hole?\n");

        for(int j =0; j<16; j++){
        check_table2[i].data[j]=0xdeadbeef;
        }
       // printf("host access : 0x%x\n", p_table[access_script[i]] );
    }
    printf("finish initialing host\n");
    printf("orambucket size %d \n",sizeof(OramB));
    uint16_t* cup_table;
    uint16_t* cucheck_table;
    TDBlock<BLOCKSIZE>* cucheck_table2;
    uint32_t* cuaccess_script;
   // uint32_t* cuorampath;
    OramB* cuoramtree;
    OramD* cudoramtree; 
    cudaError_t pterr = cudaMalloc((void**)&cup_table,sizeof(uint16_t) *( 1<<BLOCKNUMLOG));
    if(pterr != cudaSuccess){
     printf("The pterror is %s", cudaGetErrorString(pterr));
    }
    cudaError_t err = cudaMalloc((void**)&cucheck_table,sizeof(uint16_t)*(ACCESSNUM));
    if(err != cudaSuccess){
     printf("The error is %s", cudaGetErrorString(err));
    }
    cudaError_t errr = cudaMalloc((void**)&cucheck_table2,sizeof(TDBlock<BLOCKSIZE>)*(ACCESSNUM));
    if(errr != cudaSuccess){
     printf("The error2 is %s", cudaGetErrorString(errr));
    }
    cudaMalloc((void**)&cuaccess_script,sizeof(uint32_t) *(ACCESSNUM));
   // cudaMalloc((void**)&cuorampath,sizeof(uint32_t) *( 1<<LEAFNUMLOG));
    cudaMalloc((void**)&cuoramtree,sizeof(OramB) *( TREESIZE));
    cudaMalloc((void**)&cudoramtree,sizeof(OramD) *( TREESIZE));
    
    cudaError_t pterr2 = cudaMemcpy(cup_table, p_table, (1<<BLOCKNUMLOG) * sizeof(uint16_t),cudaMemcpyHostToDevice);
    if(pterr2 != cudaSuccess){
     printf("The pt copy htom error is %s", cudaGetErrorString(pterr2));
    }
   
    cudaMemcpy(cuaccess_script, access_script, (ACCESSNUM) * sizeof(uint32_t),cudaMemcpyHostToDevice);
   // cudaMemcpy(cuorampath, orampath, (1<<LEAFNUMLOG) * sizeof(uint32_t),cudaMemcpyHostToDevice);
    cudaMemcpy(cuoramtree, oramtree, (TREESIZE) * sizeof(OramB),cudaMemcpyHostToDevice);
    cudaMemcpy(cudoramtree, doramtree, (TREESIZE) * sizeof(OramD),cudaMemcpyHostToDevice);
    CUDA_CALL(cudaMemcpy(cucheck_table2, check_table2,(ACCESSNUM)*sizeof(TDBlock<BLOCKSIZE>), cudaMemcpyHostToDevice));
    oramshare<<<CUDABLOCKNUM,CUDATHREADNUM>>>(cup_table,cuaccess_script,cucheck_table, cuoramtree, cucheck_table2, devStates,cudoramtree);
    if (cudaPeekAtLastError() != cudaSuccess) {
    	printf("The peek last error is %s", cudaGetErrorString(cudaGetLastError()));
    }
    cudaDeviceSynchronize();
   // CUDA_CALL(cudaMemcpy(doramtree,cudoramtree,(TREESIZE)*sizeof(OramD),cudaMemcpyDeviceToHost));
    cudaError_t err2 = cudaMemcpy(check_table, cucheck_table, (ACCESSNUM) * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    if(err2 != cudaSuccess){
     printf("after  checktable copy error is %s\n", cudaGetErrorString(err2));
    }
    cudaError_t err3 = cudaMemcpy(check_table2, cucheck_table2, (ACCESSNUM) * sizeof(TDBlock<BLOCKSIZE>), cudaMemcpyDeviceToHost);
    if(err3 != cudaSuccess){
     printf("after  checktable copy error is %s\n", cudaGetErrorString(err3));
    }
    printf("gpu finished\n");
    bool pass = true; 
	bool dpass = true; 
    for (int i =0 ; i< ACCESSNUM ; i++){
		if (check_table[i] != access_script[i]){
		//	printf("fail test, access number: %d\n", i);
			pass = false; 
		}

		if( check_table2[i] != resultlist[access_script[i]]){
			dpass = false;
                       printf("access data %d not correct\n",i);
                      for(int j = 0; j< 16;j++){
                      printf("expected: %x, actual %x\n",resultlist[access_script[i]].data[j], check_table2[i].data[j]);
                      }
		}
      // int bucketindex = (1<<LEAFNUMLOG) - 1 + p_table[access_script[i]]; 
       //printf ("bucket index %d \n", bucketindex);
       /*for(int j = 11 ; j >= 0; j--) {
         
        if (check_table[i*24+j*2] !=  oramtree[bucketindex].id[0]){
            pass = false; 
            printf("fail 0 id: 0x%x 0x%x real id 0x%X,  0x%x\n" ,bucketindex,oramtree[bucketindex].id[0],check_table2[i*24+j*2], check_table[i*24+j*2] );
        }
        else
        { 
            printf("pass 0 id: 0x%x 0x%x real id 0x%x, 0x%x\n" ,bucketindex,oramtree[bucketindex].id[0],check_table2[i*24+j*2] ,check_table[i*24+j*2] );
        }
        if (check_table[i*24+j*2+1] !=  oramtree[bucketindex].id[1]){
            pass = false;
            printf("fail 1 id: 0x%x 0x%x real id 0x%x 0x%x\n" ,bucketindex, oramtree[bucketindex].id[1], check_table2[i*24+j*2+1],check_table[i*24+j*2+1] );
           
        }
        else
        { 
            printf("pass 0 id: 0x%x 0x%x real id 0x%x 0x%x\n" ,bucketindex,oramtree[bucketindex].id[0],check_table2[i*24+j*2+1] ,check_table[i*24+j*2] );
        }
         bucketindex = (bucketindex-1)/2; 

       }*/
	}
    
    printf("\nfinished \n");
    if (pass) {
		printf("All meta data correct\n");
	}else{
		printf("Some meta data not correct\n");
	}

	 if (dpass) {
		printf("All data correct\n");
	}else{
		printf("Some data not correct\n");
	}
    
    cudaFree(cuaccess_script);
    cudaFree(cup_table);
    cudaFree(cucheck_table);
    cudaFree(cucheck_table2);
    //cudaFree(cuorampath);
    cudaFree(cuoramtree);
    cudaFree(cudoramtree);
    cudaFree(devStates);
    delete[] p_table;
    delete[] access_script;
    //delete[] orampath;
    delete[] check_table; 
    delete[] check_table2; 
    delete[] oramtree; 
    delete[] doramtree; 
	delete[] resultlist; 
    cudaDeviceReset();
    return 0;
}
