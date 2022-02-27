simulation: simulation-cuda.cu main.cpp ElectricFieldCuda.cu ElectricFieldCpu.cpp Defines.cpp
	nvcc main.cpp simulation-cuda.cu ElectricFieldCuda.cu ElectricFieldCpu.cpp Defines.cpp -o pe-simulation -I/opt/cuda/samples/common/inc -lGL -lGLU -lglut -lGLEW  -lm --compiler-options -Wall

sm_50: simulation-cuda.cu main.cpp ElectricFieldCuda.cu ElectricFieldCpu.cpp Defines.cpp
	nvcc -arch=sm_50 main.cpp simulation-cuda.cu ElectricFieldCuda.cu ElectricFieldCpu.cpp Defines.cpp -o pe-simulation -I/opt/cuda/samples/common/inc -lGL -lGLU -lglut -lGLEW  -lm --compiler-options -Wall

clean:
	rm pe-simulation

.PHONY:
	clean
