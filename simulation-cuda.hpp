#ifndef PE_SIMULATION_CUDA_HPP
#define PE_SIMULATION_CUDA_HPP

#include "ElectricFieldCuda.hpp"

void cudaUpdateTexture(ElectricFieldCuda * electricField, float dt);
void cudaRegisterPBO();

#endif