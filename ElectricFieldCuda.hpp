#ifndef ELECTRIC_FIELD_CUDA_HPP
#define ELECTRIC_FIELD_CUDA_HPP

#include "ElectricField.hpp"

class ElectricFieldCuda : public ElectricField {
    public:

    // indices of cells
    int * cIndices;
    // indices of particles
    int * pIndices;

    ElectricFieldCuda(int p_count, int width, int height);

    void generateParticleData(int seed);
    void updateTexture(float dt);

    ~ElectricFieldCuda();
};

#endif