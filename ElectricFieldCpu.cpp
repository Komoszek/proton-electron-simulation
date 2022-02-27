#include <GL/glew.h>
#include <GL/freeglut.h>

#include "ElectricFieldCpu.hpp"
#include "simulation-cpu.hpp"
#include <random>

ElectricFieldCpu::ElectricFieldCpu(int p_count, int width, int height){
    this->p_count = p_count;
    this->width = width;
    this->height = height;

    p_x = new float[this->p_count];
    p_y = new float[this->p_count];
    p_c = new float[this->p_count];
    p_m = new float[this->p_count];

    p_Vx = new float[this->p_count];
    p_Vy = new float[this->p_count];

    cpIndices = new std::pair<int, int>[this->p_count];

    cIndices_start = new int[GRID_SIZE];
    cIndices_end = new int[GRID_SIZE];

    int texture_size = this->width * this->height;

    potential_x = new float[texture_size];
    potential_y = new float[texture_size];

    // int grid_size = GRID_SIZE * sizeof(float);
    
    int grid_size = (GRID_SIZE + MAJOR_GRID_SIZE);

    // TODO SPLIT INTO TWO
    uniform_grid_p_x = new float[grid_size];
    uniform_grid_p_y = new float[grid_size];
    uniform_grid_p_c = new float[grid_size];

    uniform_grid_n_x = new float[grid_size];
    uniform_grid_n_y = new float[grid_size];
    uniform_grid_n_c = new float[grid_size];

    #ifdef MEASURE_TIME

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::system_clock> stop;

    #endif

    generateParticleData(RANDOM_SEED);

    #ifdef MEASURE_TIME
    
    stop = std::chrono::high_resolution_clock::now();
    dataGenerateTime = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;;

    #endif

}

void ElectricFieldCpu::generateParticleData(int seed) {
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<> distrPos(0.25f, 0.75f);
    std::uniform_real_distribution<> distrSpeed(-40.0f, 40.0f);
    std::uniform_int_distribution<> distBin(0, 1);

    for(int i = 0; i < p_count; i++){
        p_x[i] = distrPos(gen) * width;
    }

    for(int i = 0; i < p_count; i++){
        p_y[i] = distrPos(gen) * height;
    }

    for(int i = 0; i < p_count; i++){
        p_Vx[i] = distrSpeed(gen);
    }


    for(int i = 0; i < p_count; i++){
        p_Vy[i] = distrSpeed(gen);
    }

    for(int i = 0; i < p_count; i++){
        if (distBin(gen)) {
            p_c[i] = PROTON_CHARGE;
            p_m[i] = PROTON_MASS;
        } else {
            p_c[i] = ELECTRON_CHARGE;
            p_m[i] = ELECTRON_MASS;
        }
    }
}

ElectricFieldCpu::~ElectricFieldCpu(){
    delete [] p_x;
    delete [] p_y;
    delete [] p_c;
    delete [] p_m;

    delete [] p_Vx;
    delete [] p_Vy;

    delete [] potential_x;
    delete [] potential_y;

    delete [] uniform_grid_p_x;
    delete [] uniform_grid_p_y;
    delete [] uniform_grid_p_c;

    delete [] uniform_grid_n_x;
    delete [] uniform_grid_n_y;
    delete [] uniform_grid_n_c;

    delete [] cpIndices;
    delete [] cIndices_end;
    delete [] cIndices_start;
}

void ElectricFieldCpu::updateTexture(float dt) {
    UpdateTextureCPU(this, dt);
}