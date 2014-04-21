#include<stdio.h>
#include<stdint.h>

template<int B>
class TBucket {
public:
  //uint16_t slotavail;
  uint16_t id[B];
  void init(int a){
    for(int i=0; i< B; i++){
       id[i] = ( 0x8000 |(a++));
    }
  }    
  
  void initzero(void){
    for(int i=0; i< B; i++){
       id[i] = 0;
    }
  }    
  

};

template <int SIZE>
class TDBlock {

public:
  uint32_t data[SIZE/4];
  
  void init(void){
     for (int i=0; i< SIZE/4; i++){
         data[i] = rand();         
     }

  }
  TDBlock(void) {}; 
  TDBlock(const TDBlock<SIZE>& that){
	  for (int i=0; i< SIZE/4; i++){
         data[i] = that.data[i];         
     }
  }
  TDBlock<SIZE>& operator= (const TDBlock<SIZE>& that){
	  for (int i=0; i< SIZE/4; i++){
         data[i] = that.data[i];         
     }
	  return *this;
  }

  bool operator==(const TDBlock<SIZE>& that ){
	  bool temp = true; 
	  for (int i=0; i< SIZE/4; i++){
         if(data[i]!= that.data[i]){

			  temp = false; 
		 }
     }
	  return temp ; 
  }

  bool operator!=(const TDBlock<SIZE>& that ){
	 
	  return !(*this==that) ; 
  }
};


template <int B, int SIZE>
class TDBucket{

public:
  TDBlock<SIZE> block[B];
  void init(void){
     for (int i=0; i< B; i++){
         block[i].init();         
     }

  }

};



