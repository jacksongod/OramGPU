include ./make.config

CC := nvcc
NVCCFLAG :=   -arch=compute_35 -code=sm_35 -lineinfo --machine 64 -Xcompiler -rdynamic
PTXASFLAG:= --ptxas-options=-v -Xptxas -dlcm=cg
INCLUDE := ../cub

VARIABLE :=# -DCUDA_ARCH=350
SRC = kernel.cu 
EXE = oramgpu

release: $(SRC)  bucket.h
	$(CC) $(NVCCFLAG) $(PTXASFLAG) $(VARIABLE)$(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm oramgpu
