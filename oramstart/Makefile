include ./make.config

CC := nvcc
NVCCFLAG :=  -O2 --ptxas-options=-v -Xptxas -dlcm=cg  -arch=compute_35 -code=sm_35 -lineinfo --machine 64

INCLUDE := ./

SRC = kernel.cu 
EXE = oramgpu

release: $(SRC)
	$(CC) $(NVCCFLAG) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm oramgpu
