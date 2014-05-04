include ./make.config

CC := nvcc
NVCCFLAG :=  -O2  -arch=compute_35 -code=sm_35 -lineinfo --machine 64
PTXASFLAG:= --ptxas-options=-v -Xptxas -dlcm=cg
INCLUDE := ./

SRC = kernel.cu 
EXE = oramgpu

release: $(SRC)
	$(CC) $(NVCCFLAG) $(PTXASFLAG) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm oramgpu
