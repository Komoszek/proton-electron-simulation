#include "ElectricFieldCuda.hpp"
#include "simulation-cuda.hpp"
#include <random>

ElectricFieldCuda::ElectricFieldCuda(int p_count, int width, int height){
    this->p_count = p_count;
    this->width = width;
    this->height = height;

    int p_size = this->p_count * sizeof(float);

    cudaMalloc(&p_x, p_size);
    cudaMalloc(&p_y, p_size);
    cudaMalloc(&p_c, p_size);
    cudaMalloc(&p_m, p_size);

    cudaMalloc(&p_Vx, p_size);
    cudaMalloc(&p_Vy, p_size);

    cudaMalloc(&cIndices, this->p_count*sizeof(int));
    cudaMalloc(&pIndices, this->p_count*sizeof(int));

    cudaMalloc(&cIndices_start, GRID_SIZE * sizeof(int));
    cudaMalloc(&cIndices_end, GRID_SIZE * sizeof(int));

    int texture_size = this->width * this->height * sizeof(float);
    cudaMalloc(&potential_x, texture_size);
    cudaMalloc(&potential_y, texture_size);

    cudaMemset(potential_x, 0, texture_size);
    cudaMemset(potential_y, 0, texture_size);


    // int grid_size = GRID_SIZE * sizeof(float);
    
    // WORKAROUND so i don't have to add new variables - have to change
    int grid_size = (GRID_SIZE + MAJOR_GRID_SIZE) * sizeof(float);

    // TODO SPLIT INTO TWO
    cudaMalloc(&uniform_grid_p_x, grid_size);
    cudaMalloc(&uniform_grid_p_y, grid_size);
    cudaMalloc(&uniform_grid_p_c, grid_size);

    cudaMalloc(&uniform_grid_n_x, grid_size);
    cudaMalloc(&uniform_grid_n_y, grid_size);
    cudaMalloc(&uniform_grid_n_c, grid_size);

    #ifdef MEASURE_TIME

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start);

    #endif

    generateParticleData(RANDOM_SEED);

    #ifdef MEASURE_TIME

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    dataGenerateTime += milliseconds;

    #endif
    
    cudaRegisterPBO();
}

void ElectricFieldCuda::generateParticleData(int seed) {
    int p_size = p_count * sizeof(float);
    std::mt19937 gen(RANDOM_SEED);
    std::uniform_real_distribution<> distrPos(0.25f, 0.75f);
    std::uniform_real_distribution<> distrSpeed(-40.0f, 40.0f);
    std::uniform_int_distribution<> distBin(0, 1);

    float * tempArr = new float[p_count];
    float * tempArr2 = new float[p_count];

    for(int i = 0; i < p_count; i++){
        tempArr[i] = distrPos(gen) * width;
    }

    #ifdef MEASURE_TIME

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    #endif

    cudaMemcpy(p_x, tempArr, p_size, cudaMemcpyHostToDevice);

    #ifdef MEASURE_TIME

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    memcpyTime += milliseconds;

    #endif


    for(int i = 0; i < p_count; i++){
        tempArr[i] = distrPos(gen) * height;
    }

    #ifdef MEASURE_TIME
    
    cudaEventRecord(start);

    #endif

    cudaMemcpy(p_y, tempArr, p_size, cudaMemcpyHostToDevice);

    #ifdef MEASURE_TIME

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    memcpyTime += milliseconds;

    #endif

    for(int i = 0; i < p_count; i++){
        tempArr[i] = distrSpeed(gen);
    }

    #ifdef MEASURE_TIME
    
    cudaEventRecord(start);

    #endif

    cudaMemcpy(p_Vx, tempArr, p_size, cudaMemcpyHostToDevice);

    #ifdef MEASURE_TIME

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    memcpyTime += milliseconds;

    #endif

    for(int i = 0; i < p_count; i++){
        tempArr[i] = distrSpeed(gen);
    }

    #ifdef MEASURE_TIME
    
    cudaEventRecord(start);

    #endif

    cudaMemcpy(p_Vy, tempArr, p_size, cudaMemcpyHostToDevice);

    #ifdef MEASURE_TIME

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    memcpyTime += milliseconds;

    #endif

    for(int i = 0; i < p_count; i++){
        if (distBin(gen)) {
            tempArr[i] = PROTON_CHARGE;
            tempArr2[i] = PROTON_MASS;
        } else {
            tempArr[i] = ELECTRON_CHARGE;
            tempArr2[i] = ELECTRON_MASS;
        }
    }

    #ifdef MEASURE_TIME
    
    cudaEventRecord(start);

    #endif

    cudaMemcpy(p_c, tempArr, p_size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_m, tempArr2, p_size, cudaMemcpyHostToDevice);

    #ifdef MEASURE_TIME

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    memcpyTime += milliseconds;

    #endif


    delete [] tempArr;
    delete [] tempArr2;
}

ElectricFieldCuda::~ElectricFieldCuda(){
    cudaFree(p_x);
    cudaFree(p_y);
    cudaFree(p_c);
    cudaFree(p_m);

    cudaFree(p_Vx);
    cudaFree(p_Vy);

    cudaFree(potential_x);
    cudaFree(potential_y);

    cudaFree(uniform_grid_p_x);
    cudaFree(uniform_grid_p_y);
    cudaFree(uniform_grid_p_c);

    cudaFree(uniform_grid_n_x);
    cudaFree(uniform_grid_n_y);
    cudaFree(uniform_grid_n_c);


    cudaFree(cIndices);
    cudaFree(pIndices);
    cudaFree(cIndices_end);
    cudaFree(cIndices_start);
}

void ElectricFieldCuda::updateTexture(float dt) {
    cudaUpdateTexture(this, dt);    
}