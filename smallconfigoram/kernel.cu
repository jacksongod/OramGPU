
// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "bucket.h"
#include "aesctr.h"
#include <curand_kernel.h>
#include <cuda.h>

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

#define ACCESSNUM 10
/**
 * Main
 */
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

typedef TBucket<BLOCKPERBUCKET> OramB; 
typedef TDBucket<BLOCKPERBUCKET,BLOCKSIZE> OramD;
typedef TDBlock<BLOCKSIZE> OramBlock;

__device__ __forceinline__ void aes_fround (int secondposition, int thirdposition,int fourthposition, int& outword, int& inword,uint32_t RK,
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
__device__ __forceinline__ void aes_finalfround (int secondposition, int thirdposition,int fourthposition, int& outword, int& inword,uint32_t RK,
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



__device__ __forceinline__ int calcindex(int level, uint16_t path){
    return (path>>(LEAFNUMLOG-level))+ (1<<level) -1 ;

}

__global__ void oramshare(uint16_t* position_table, 
                          uint32_t* access_script,
                          uint16_t* checktable, 
                          OramB* oramtree, 
                          TDBlock<BLOCKSIZE>* checktable2,
                          curandState *randstate,
                          OramD* datatree,
                          aes_context *ctx,
                                    const uint32_t* __restrict__ RDFT0,
                                    const uint32_t* __restrict__ RDFT1,
                                    const uint32_t* __restrict__ RDFT2,
                                    const uint32_t* __restrict__ RDFT3,
                                    const unsigned char* __restrict__ RDFSb
						){
	int tid =  threadIdx.x;
        curandState localrandState = randstate[tid];
        __shared__ OramB metatree[TREESIZE];              //4K*sizeof(OramB) = 16KB
       // __shared__ OramB treepath[LEAFNUMLOG+1];          //12*sizeof(OramB) = 48B
        __shared__ uint32_t treepathlock[(LEAFNUMLOG+1)*2];    //96B
        __shared__ uint32_t streepathlock[STASHSIZE];    //1kB
        __shared__ uint16_t stash [STASHSIZE];            //256B
        __shared__ uint32_t expectedblockindex; 
        __shared__ uint32_t stashlock[STASHSIZE];          //1kB
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
       __shared__ aes_context key; 
       __shared__ uint32_t returnblock[BLOCKSIZE/4];
    //copy position table from global memory to shared memory
    //   localtable[tid*MAPSIZEPERTHREAD] = position_table[tid*MAPSIZEPERTHREAD];
     //  localtable[tid*MAPSIZEPERTHREAD+1] = position_table[tid*MAPSIZEPERTHREAD+1];
     //  localtable[tid*MAPSIZEPERTHREAD+2] = position_table[tid*MAPSIZEPERTHREAD+2];
     //  localtable[tid*MAPSIZEPERTHREAD+3] = position_table[tid*MAPSIZEPERTHREAD+3];
     //localtable[tid*2] = (position_table[tid*MAPSIZEPERTHREAD+1]<<16) | position_table[tid*MAPSIZEPERTHREAD];
     //localtable[tid*2+1] = (position_table[tid*MAPSIZEPERTHREAD+3]<<16) | position_table[tid*MAPSIZEPERTHREAD+2];
 
    memcpy(&localtable[tid*MAPSIZEPERTHREAD],&position_table[tid*MAPSIZEPERTHREAD],sizeof(uint16_t)*MAPSIZEPERTHREAD);
    // copy metadata tree from global memory to shared memory
    memcpy(&metatree[tid*MAPSIZEPERTHREAD], &oramtree[tid*MAPSIZEPERTHREAD],sizeof(OramB)*MAPSIZEPERTHREAD);
   if (tid <256)  stashlock[tid] = 0;
  // if (tid< 24) treepathlock[tid] = 0;
  // if (tid<12) streepathlock[tid] = 0;
   //if (tid ==256) pathcount = 24;
   if (tid ==0) { 
        stashcount = STASHSIZE;
        maxstashcount = 256;
        key = *ctx;
   }
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
    int startindex= tid*8 ; //tid *8
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
         if (tid< 24){  //copy entire path to local registers (12 levels, 24 blocks)
               treepathlock[tid] = 0;
                //pathcount = 24; 
            //printf("rand : %d is %d\n",tid,  (unsigned)(curand(&localrandState))%(1<<LEAFNUMLOG));
           int treeindex = calcindex(tid/2, pathid);
           //printf("id: %d, index %d\t",tid,treeindex );
           uint16_t id = metatree[treeindex].id[tid%2];
		   metatree[treeindex].id[tid%2] = 0;
           if((id>>15) == 1){   // if data is valid 
            // printf("id: %d valid data \t ", tid); 
              while(true){
                 if (!atomicCAS(&stashlock[startindex],0,1 )){
            //         printf("id: %d, foundposition\t",tid);
                    stashaccessloc[tid] = startindex;
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
			  }
            //   printf("out\n");
		   }else{
			   stashaccessloc[tid] = 999;
		   }
		   
                     
         }  

        // data decryption 
         int myfetcheddata; 
         int initxword; 
         int bucketid = tid/16;
         int treeindex = calcindex(bucketid/2, pathid);
         int laneid = tid%32; 
         int whichword = tid%4;    
         int whichdata = tid%16;
         if (tid < 384){
            int xword, yword;  
           // int whichoramblock = tid/16;
           // int whichciphperblock = tid/4%4;
            uint32_t* RK_ptr = key.buf; 
            //OramBlock tempblock = datatree[treeindex].block[bucketid%2];
            //initxword = xword = *((int*)&tempblock.nonce[whichword*4]);
            //int encrypteddata = tempblock.data[whichdata]; 
            
            initxword = xword = *((int*)&datatree[treeindex].block[bucketid%2].nonce[whichword*4]);
            //to do ..increment  nonce for different cipher blocks
            int encrypteddata = datatree[treeindex].block[bucketid%2].data[whichdata]; 
            xword ^= *(RK_ptr+whichword);
            	RK_ptr += 4; 
	int secondposition = (whichword+1)%4-whichword+laneid;
	int thirdposition = (whichword+2)%4-whichword+laneid; 
	int fourthposition = (whichword+3)%4-whichword+laneid;
	for( int i = (key.nr >> 1) - 1; i > 0; i-- )
        {
            aes_fround( secondposition,thirdposition,fourthposition, yword, xword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);
			RK_ptr += 4; 
            aes_fround( secondposition,thirdposition,fourthposition, xword, yword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);
			RK_ptr += 4; 
        }
            aes_fround( secondposition,thirdposition,fourthposition, yword, xword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);

		RK_ptr += 4; 
            aes_finalfround( secondposition,thirdposition,fourthposition, xword, yword,*(RK_ptr+whichword),RDFSb);

           myfetcheddata = xword ^ encrypteddata; 
         }
         __syncthreads();
           if (tid<384){
		if(stashaccessloc[bucketid] !=999){ //999 means this block in the tree path is empty
              		datastash[(stashaccessloc[bucketid]%STASHSIZE)*16+whichdata] = myfetcheddata; 
		}

         }
		
         if (tid < STASHSIZE){
             if (stashlock[tid]!=0){
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
                       stashlock[tid] = 0;
                //       atomicAdd(&stashcount,1);
                    //   atomicSub(&pathcount,1);
                       break; 
                   } else if (!atomicCAS(&treepathlock[(level<<1)+1],0,1)){
                       writebackloc[blockloc+1] = tid; 
                       metatree[treeindex].id[1] = 0x8000|myblockid;
                       stashlock[tid] = 0;
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
         //data encryption 
         int mywbdata;
	 if (tid < 384){
            int xword, yword;  
           // int whichoramblock = tid/16;
           // int whichciphperblock = tid/4%4;
            uint32_t* RK_ptr = key.buf; 
            //OramBlock tempblock = datatree[treeindex].block[bucketid%2];
            //initxword = xword = *((int*)&tempblock.nonce[whichword*4]);
            //int encrypteddata = tempblock.data[whichdata]; 
            
            xword = *((int*)&datatree[treeindex].block[bucketid%2].nonce[whichword*4]);
            //to do ..increment  nonce for different cipher blocks
            int decrypteddata = datastash[(writebackloc[bucketid]%STASHSIZE)*16+whichdata]; 
            xword ^= *(RK_ptr+whichword);
            	RK_ptr += 4; 
	int secondposition = (whichword+1)%4-whichword+laneid;
	int thirdposition = (whichword+2)%4-whichword+laneid; 
	int fourthposition = (whichword+3)%4-whichword+laneid;
	for( int i = (key.nr >> 1) - 1; i > 0; i-- )
        {
            aes_fround( secondposition,thirdposition,fourthposition, yword, xword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);
			RK_ptr += 4; 
            aes_fround( secondposition,thirdposition,fourthposition, xword, yword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);
			RK_ptr += 4; 
        }
            aes_fround( secondposition,thirdposition,fourthposition, yword, xword,*(RK_ptr+whichword), RDFT0,RDFT1,RDFT2,RDFT3);

		RK_ptr += 4; 
            aes_finalfround( secondposition,thirdposition,fourthposition, xword, yword,*(RK_ptr+whichword),RDFSb);

            mywbdata = xword ^ decrypteddata; 
         }



 
         __syncthreads();

         // writeback data back from stash to tree 
         if (tid < 384){
           //int bucketid = tid/16;
           //  int treeindex = calcindex(bucketid/2, pathid);
           //   int whichdata = tid%16;
           //   int whichblock = bucketid%2;
			
             //datatree[treeindex].block[bucketid%2].data[whichdata] = datastash[(writebackloc[bucketid]%STASHSIZE)*16+whichdata];       
             datatree[treeindex].block[bucketid%2].data[whichdata] = mywbdata;       
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
    // AES initialization
    unsigned char key[16];
    aes_context ctx; 
    int len = 16;
    memcpy( key, aes_test_ctr_key[0], 16 );
     aes_context::aes_setkey_enc( &ctx, key);
    
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

	CUDA_CALL(cudaMalloc(&dctx, sizeof(aes_context)));

	CUDA_CALL(cudaMemcpy(dctx,&ctx,sizeof(aes_context),cudaMemcpyHostToDevice));
	 CUDA_CALL(cudaMemcpy(  RDFT0,  FT0,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice  ));
	 CUDA_CALL(cudaMemcpy(  RDFT1,  FT1,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice  ));
	 CUDA_CALL(cudaMemcpy(  RDFT2,  FT2,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice  ));
	 CUDA_CALL(cudaMemcpy(  RDFT3,  FT3,   sizeof(uint32_t)*256,cudaMemcpyHostToDevice ));
	 CUDA_CALL(cudaMemcpy(  RDFSb,  FSb,   sizeof(char)*256, cudaMemcpyHostToDevice  ));



   // GPU ORAM pointers initialization  
    uint16_t* cup_table;
    uint16_t* cucheck_table;
    TDBlock<BLOCKSIZE>* cucheck_table2;
    uint32_t* cuaccess_script;
   // uint32_t* cuorampath;
    OramB* cuoramtree;
    OramD* cudoramtree; 
    CUDA_CALL(cudaMalloc((void**)&cup_table,sizeof(uint16_t) *( 1<<BLOCKNUMLOG)));
    CUDA_CALL(cudaMalloc((void**)&cucheck_table,sizeof(uint16_t)*(ACCESSNUM)));
    CUDA_CALL(cudaMalloc((void**)&cucheck_table2,sizeof(TDBlock<BLOCKSIZE>)*(ACCESSNUM)));
    CUDA_CALL(cudaMalloc((void**)&cuaccess_script,sizeof(uint32_t) *(ACCESSNUM)));
   // cudaMalloc((void**)&cuorampath,sizeof(uint32_t) *( 1<<LEAFNUMLOG));
    CUDA_CALL(cudaMalloc((void**)&cuoramtree,sizeof(OramB) *( TREESIZE)));
    CUDA_CALL(cudaMalloc((void**)&cudoramtree,sizeof(OramD) *( TREESIZE)));
    
    CUDA_CALL(cudaMemcpy(cup_table, p_table, (1<<BLOCKNUMLOG) * sizeof(uint16_t),cudaMemcpyHostToDevice));
   
    CUDA_CALL(cudaMemcpy(cuaccess_script, access_script, (ACCESSNUM) * sizeof(uint32_t),cudaMemcpyHostToDevice));
   // cudaMemcpy(cuorampath, orampath, (1<<LEAFNUMLOG) * sizeof(uint32_t),cudaMemcpyHostToDevice);
    CUDA_CALL(cudaMemcpy(cuoramtree, oramtree, (TREESIZE) * sizeof(OramB),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(cudoramtree, doramtree, (TREESIZE) * sizeof(OramD),cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(cucheck_table2, check_table2,(ACCESSNUM)*sizeof(TDBlock<BLOCKSIZE>), cudaMemcpyHostToDevice));
    oramshare<<<CUDABLOCKNUM,CUDATHREADNUM>>>(cup_table,cuaccess_script,cucheck_table, cuoramtree, cucheck_table2, devStates,cudoramtree,dctx,RDFT0,RDFT1,RDFT2,RDFT3,RDFSb);
    cudaDeviceSynchronize();
   // CUDA_CALL(cudaMemcpy(doramtree,cudoramtree,(TREESIZE)*sizeof(OramD),cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(check_table, cucheck_table, (ACCESSNUM) * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(check_table2, cucheck_table2, (ACCESSNUM) * sizeof(TDBlock<BLOCKSIZE>), cudaMemcpyDeviceToHost));
    printf("gpu finished\n");
    bool pass = true; 
	bool dpass = true; 
    for (int i =0 ; i< ACCESSNUM ; i++){
		if (check_table[i] != access_script[i]){
		//	printf("fail test, access number: %d\n", i);
			pass = false; 
		}

	/*	if( check_table2[i] != resultlist[access_script[i]]){
			dpass = false;
                       printf("access data %d not correct\n",i);
                      for(int j = 0; j< 16;j++){
                      printf("expected: %x, actual %x\n",resultlist[access_script[i]].data[j], check_table2[i].data[j]);
                      }
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
        cudaFree(dctx);
        cudaFree(RDFT0);
        cudaFree(RDFT1);
        cudaFree(RDFT2);
        cudaFree(RDFT3);
        cudaFree(RDFSb);
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
