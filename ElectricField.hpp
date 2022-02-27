#ifndef ELECTRIC_FIELD_HPP
#define ELECTRIC_FIELD_HPP

#include "Defines.hpp"

class ElectricField {
    public:

    // particles positions (x, y), charges (c) and masses (m)
    float * p_x;
    float * p_y;
    float * p_c;
    float * p_m;

    // particles velocities (Vx, Vy)
    float * p_Vx;
    float * p_Vy;

    // Number of particles
    int p_count;

    // Electric field potentials on pixels
    float * potential_x;
    float * potential_y;

    // Virtual particles on uniform grid
    // positive particles
    float * uniform_grid_p_x;
    float * uniform_grid_p_y;
    float * uniform_grid_p_c;
    // negative particles
    float * uniform_grid_n_x;
    float * uniform_grid_n_y;
    float * uniform_grid_n_c;

    // Electric field (texture) width and height
    int width;
    int height;

    // starting indices of particles in cells; i-th value is index of first particle in i-th cell
    int * cIndices_start;
    // ending indices of particles in cells; i-th value is index of last particle in i-th cell
    int * cIndices_end;

    virtual void generateParticleData(int seed) = 0;
    virtual void updateTexture(float dt) = 0;
    virtual ~ElectricField() {};
    
};

#endif