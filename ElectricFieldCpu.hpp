#ifndef ELECTRIC_FIELD_CPU_HPP
#define ELECTRIC_FIELD_CPU_HPP

#include "Defines.hpp"
#include "ElectricField.hpp"
#include <utility>

class ElectricFieldCpu : public ElectricField {
    public:

    // (cellID, particleID) pairs
    std::pair<int, int> * cpIndices;

    // texture pixel data
    unsigned char * pixels;

    ElectricFieldCpu(int p_count, int width, int height);

    void generateParticleData(int seed);
    void updateTexture(float dt);

    ~ElectricFieldCpu();
};

#endif