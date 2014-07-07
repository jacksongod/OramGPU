include ./make.config

CC := nvcc
NVCCFLAG :=  -O2  -arch=compute_35 -code=sm_35 -lineinfo --machine 64
PTXASFLAG:= --ptxas-options=-v -Xptxas -dlcm=cg
INCLUDE := ../cub
VARIABLE :=# -DCUDA_ARCH=350
SRC = kernel2.cu#example_block_radix_sort.cu 
EXE = radixsort 

release: $(SRC) aesctr.h
	 $(CC) $(NVCCFLAG) $(PTXASFLAG) $(VARIABLE)  $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm aesgpu
