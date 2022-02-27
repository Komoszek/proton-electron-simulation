#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>

#include "ElectricFieldCuda.hpp"
#include "Defines.hpp"
#include "simulation-cuda.hpp"

static struct cudaGraphicsResource* cuda_pbo_resource;

// Calculates index of cell that position belongs to 
__device__ int posToSubIndices(float x, float dim){
    int p = (x / dim) * GRID_DIM;
    
    return p < GRID_DIM ? p : GRID_DIM - 1;
}

// Calculates electric field potential on (x, y) created by particle 
__device__ void CalculateEF(float p_x, float p_y, float p_c, float x, float y, float &ef_x, float &ef_y, float &ef_potential){
        float dx = x - p_x;
        float dy = y - p_y;

        float distanceInvSqrt = rsqrtf(dx*dx + dy*dy + SOFTENING);
        
        ef_potential = K_E * p_c * distanceInvSqrt * distanceInvSqrt;

        ef_x = ef_potential * dx * distanceInvSqrt;
        ef_y = ef_potential * dy * distanceInvSqrt;
}

// Clamps value v to [min, max]
__device__ float clamp(float v, float min, float max){
  return fminf(fmaxf(v, min), max);
}

// calculates
__device__ __host__ int ceilPosInt(int a, int b){
    return  1 + (a - 1) / b;
}

// Calculates electric field potential on (x, y) created by all particles referenced between pStart-th and pEnd-th position in pIndices array
__device__ void CalculateEFFromGrid(float * p_x, float * p_y, float * p_c, float x, float y, int pStart, int pEnd, 
                                    int * pIndices, float &ef_x, float &ef_y, float &ef_potential){

    float temp_ef_x = 0.0f, temp_ef_y  = 0.0f, temp_ef_potential  = 0.0f;
    
    for(int p = pStart; p < pEnd; p++){
            int pIndex = pIndices[p];
            CalculateEF(p_x[pIndex], p_y[pIndex], p_c[pIndex], x, y, temp_ef_x, temp_ef_y, temp_ef_potential); 

            ef_x += temp_ef_x;
            ef_y += temp_ef_y;
            ef_potential += temp_ef_potential;
        }
}

// Calculates and draws potential created by all particles on single pixel 
__global__ void UpdateField(float p_x[], float p_y[], float p_c[], float potential_x[], float potential_y[], int cIndices_start[], 
                            int cIndices_end[], int pIndices[], int width, int height, uchar4 * d_out, float ugrid_p_x[], float ugrid_p_y[], 
                            float ugrid_p_c[], float ugrid_n_x[], float ugrid_n_y[], float ugrid_n_c[]){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int sm_pos = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    __shared__ float sm_p_x[MAJOR_GRID_SIZE + GRID_SIZE];
    __shared__ float sm_p_y[MAJOR_GRID_SIZE + GRID_SIZE];
    __shared__ float sm_p_c[MAJOR_GRID_SIZE + GRID_SIZE];

    __shared__ float sm_n_x[MAJOR_GRID_SIZE + GRID_SIZE];
    __shared__ float sm_n_y[MAJOR_GRID_SIZE + GRID_SIZE];
    __shared__ float sm_n_c[MAJOR_GRID_SIZE + GRID_SIZE];

    int tid = y * width + x;

    // Filling shared memory with virtual particles placed in each grid cell and major grid cell

    if(sm_pos < GRID_SIZE) {
        sm_p_x[sm_pos] = ugrid_p_x[sm_pos];
        sm_p_y[sm_pos] = ugrid_p_y[sm_pos];
        sm_p_c[sm_pos] = ugrid_p_c[sm_pos];

        sm_n_x[sm_pos] = ugrid_n_x[sm_pos];
        sm_n_y[sm_pos] = ugrid_n_y[sm_pos];
        sm_n_c[sm_pos] = ugrid_n_c[sm_pos];
    }

    if(sm_pos < MAJOR_GRID_SIZE){
        sm_p_x[GRID_SIZE + sm_pos] = ugrid_p_x[GRID_SIZE + sm_pos];
        sm_p_y[GRID_SIZE + sm_pos] = ugrid_p_y[GRID_SIZE + sm_pos];
        sm_p_c[GRID_SIZE + sm_pos] = ugrid_p_c[GRID_SIZE + sm_pos];

        sm_n_x[GRID_SIZE + sm_pos] = ugrid_n_x[GRID_SIZE + sm_pos];
        sm_n_y[GRID_SIZE + sm_pos] = ugrid_n_y[GRID_SIZE + sm_pos];
        sm_n_c[GRID_SIZE + sm_pos] = ugrid_n_c[GRID_SIZE + sm_pos];
    }

    __syncthreads();
    
    if(x >= width || y >= height)
      return;

    float pos_x = x + 0.5f;
    float pos_y = y + 0.5f;

    float ef_x = 0.0f, ef_y = 0.0f, ef_potential = 0.0f;
    float temp_ef_x = 0.0f, temp_ef_y = 0.0f, temp_ef_potential = 0.0f;

    // Calculate x, y cell indices and get neighboring cell rectangle
    int subind_x = posToSubIndices(pos_x, width);
    int subind_y = posToSubIndices(pos_y, height);

    int subgrid_n_minx = max(subind_x-1, 0);
    int subgrid_n_miny = max(subind_y-1, 0);

    int subgrid_n_maxx = min(subind_x+1, GRID_DIM-1);
    int subgrid_n_maxy = min(subind_y+1, GRID_DIM-1);

    int k;
    // Calculate electric field potential created by real particles from adjacent cells
    for(int j = subgrid_n_miny; j <= subgrid_n_maxy; j++ ){
        k = j * GRID_DIM + subgrid_n_minx;
        for(int i = subgrid_n_minx; i <= subgrid_n_maxx; i++, k++){
            CalculateEFFromGrid(p_x, p_y, p_c, pos_x, pos_y, cIndices_start[k], 
                            cIndices_end[k], pIndices, ef_x, ef_y, ef_potential);
        }
    }


    subgrid_n_minx /= SUBGRID_DIM;
    subgrid_n_maxx /= SUBGRID_DIM;

    subgrid_n_miny /= SUBGRID_DIM;
    subgrid_n_maxy /= SUBGRID_DIM;


    // Calculate electric field potential created by virtual major particles from non-adjacent major grid cells
    k = GRID_SIZE;
    for(int i = 0; i < MAJOR_GRID_DIM; i++){
        for(int j = 0; j < MAJOR_GRID_DIM; j++, k++){
            if(j >= subgrid_n_minx && j <= subgrid_n_maxx && i >= subgrid_n_miny && i <= subgrid_n_maxy)
                continue;
            
            if(sm_p_c[k]){
                 CalculateEF(sm_p_x[k], sm_p_y[k], sm_p_c[k], 
                    pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                ef_x += temp_ef_x;
                ef_y += temp_ef_y;
                ef_potential += temp_ef_potential;
            }
            

            if(sm_n_c[k]){
                CalculateEF(sm_n_x[k], sm_n_y[k], sm_n_c[k], 
                    pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);

                ef_x += temp_ef_x;
                ef_y += temp_ef_y;
                ef_potential += temp_ef_potential;
            }
        }
    }
    
    subgrid_n_minx *= SUBGRID_DIM;
    subgrid_n_miny *= SUBGRID_DIM;

    subgrid_n_maxx = (subgrid_n_maxx + 1) * SUBGRID_DIM;
    subgrid_n_maxy = (subgrid_n_maxy + 1) * SUBGRID_DIM;

    // Calculate electric field potential created by virtual particles from non-adjacent cells but in adjacent major cell
    for(int j = subgrid_n_miny; j < subgrid_n_maxy; j++){
        k = j * GRID_DIM + subgrid_n_minx;
        for(int i = subgrid_n_minx; i < subgrid_n_maxx; i++, k++){ 
            int d = max(abs(i -  subind_x), abs(j -  subind_y));
            if(d < 2)
                continue;

            if(sm_p_c[k]){
                 CalculateEF(sm_p_x[k], sm_p_y[k], sm_p_c[k], 
                    pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                ef_x += temp_ef_x;
                ef_y += temp_ef_y;
                ef_potential += temp_ef_potential;
            }
            

            if(sm_n_c[k]){
                CalculateEF(sm_n_x[k], sm_n_y[k], sm_n_c[k], 
                    pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                    
                ef_x += temp_ef_x;
                ef_y += temp_ef_y;
                ef_potential += temp_ef_potential;
            }
        }
    }
    
    // save x and y potentials for later calculations
    potential_x[tid] = ef_x;
    potential_y[tid] = ef_y;


    // draw potential in pixel

    ef_potential *= DRAW_COEF;

    int red = 0;
    int blue = 0;

    if(ef_potential >= 0.0f){
        red = clamp(ef_potential, 0.0f, 255.0f);
    } else {
        blue  = clamp(-ef_potential, 0.0f, 255.0f);
    }

    d_out[tid].x = red;    
    d_out[tid].y = 0;
    d_out[tid].z = blue;
    d_out[tid].w = 255;
}

// Updates particles speeds and positions using electric field potential
__global__ void UpdateParticles(float p_x[], float p_y[], float p_c[], float p_m[], float p_Vx[], float p_Vy[], float potential_x[], 
                                    float potential_y[], int p_count, int width, int height, float dt){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float temp_ef_x, temp_ef_y, temp_ef_potential;

    if(tid >= p_count)
        return;

    // Get nearest pixel indices
    int g_y = (int)p_y[tid];
    int g_x = (int)p_x[tid];
    
    if(g_x >= width){
        g_x = width -1;
    }

    if(g_y >= height){
        g_y = height -1;
    }

    // get potentials of pixel
    float ef_x = potential_x[g_y * width + g_x];
    float ef_y = potential_y[g_y * width + g_x];

    // translate grid indices to position
    float pos_x = g_x + 0.5f;
    float pos_y = g_y + 0.5f;

    // calculate electric field created by particle
    CalculateEF(p_x[tid], p_y[tid], p_c[tid], pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential); 
    
    // adjust potential by removing potential created by particle
    ef_x -= temp_ef_x;
    ef_y -= temp_ef_y;

    float pd = p_c[tid] / p_m[tid];

    // a = F / m, F = q * E => a = q * E / m
    
    float ax = ef_x * pd;
    float ay = ef_y * pd;
    
    
    p_Vx[tid] += ax * dt;
    p_Vy[tid] += ay * dt;
    
    p_x[tid] += p_Vx[tid] * dt;
    p_y[tid] += p_Vy[tid] * dt;

    if(p_x[tid] <= 0.0f){
        p_x[tid] = BORDER_EPS;
        p_Vx[tid] = -p_Vx[tid] * VELOCITY_DRAG;
    } else if(p_x[tid] >= width){
        p_x[tid] = width - BORDER_EPS;
        p_Vx[tid] = -p_Vx[tid] * VELOCITY_DRAG;
    }

    if(p_y[tid] <= 0.0f){
        p_y[tid] = BORDER_EPS;
        p_Vy[tid] = -p_Vy[tid] * VELOCITY_DRAG;
    } else if(p_y[tid] >= height){
        p_y[tid] = height - BORDER_EPS;
        p_Vy[tid] = -p_Vy[tid] * VELOCITY_DRAG;
    }
}

// Calculates cell indices for particles
__global__ void calculateIndices(float * p_x, float * p_y, int *d_out, int *d_indexes, int width, int height, int p_count){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= p_count)
        return;

    int x = posToSubIndices(p_x[tid], width);
    int y = posToSubIndices(p_y[tid], height);

    if(x >= GRID_DIM) x = GRID_DIM - 1;
    if(y >= GRID_DIM) y = GRID_DIM - 1;

    d_out[tid] = y * GRID_DIM + x;
    d_indexes[tid] = tid;
}


// Calculates virtual particles placed inside grid cells
__global__ void CalculateGridParticles(float p_x[], float p_y[], float p_c[], float ugrid_p_x[], float ugrid_p_y[], float ugrid_p_c[], 
                                    float ugrid_n_x[], float ugrid_n_y[], float ugrid_n_c[], int cIndices_start[], int cIndices_end[], int pIndices[]){
    
        int ind = blockIdx.x * blockDim.x + threadIdx.x;

        if(ind >= GRID_SIZE)
            return;

        float x_neg = 0.0f, y_neg = 0.0f, c_neg = 0.0f, x_pos = 0.0f, y_pos = 0.0f, c_pos = 0.0f;
            
        for(int p = cIndices_start[ind]; p < cIndices_end[ind]; p++){
            int pIdx = pIndices[p];

            if(p_c[pIdx] > 0.0f){
                c_pos += p_c[pIdx];
                x_pos += p_c[pIdx] * p_x[pIdx];
                y_pos += p_c[pIdx] * p_y[pIdx];
            } else if(p_c[pIdx] < 0.0f){
                c_neg += p_c[pIdx];
                x_neg += p_c[pIdx] * p_x[pIdx];
                y_neg += p_c[pIdx] * p_y[pIdx];
            }
        }

        if(c_pos != 0){
            x_pos /= c_pos;
            y_pos /= c_pos;
            ugrid_p_x[ind] = x_pos;
            ugrid_p_y[ind] = y_pos;
        } 

        ugrid_p_c[ind] = c_pos; 

        if(c_neg != 0){
            x_neg /= c_neg;
            y_neg /= c_neg;
            ugrid_n_x[ind] = x_neg;
            ugrid_n_y[ind] = y_neg;
        } 

        ugrid_n_c[ind] = c_neg;         
}

// Calculates virtual particles placed inside Major Grid Cells
__global__ void CalculateMajorGridParticles(float p_x[], float p_y[], float p_c[], float ugrid_p_x[], float ugrid_p_y[], float ugrid_p_c[], float ugrid_n_x[], 
                                            float ugrid_n_y[], float ugrid_n_c[], int pIndices[]){
        int x = threadIdx.x;
        int y = threadIdx.y;
        int tid = y * MAJOR_GRID_DIM + x;

        if(tid >= MAJOR_GRID_SIZE)
            return;

        float x_neg = 0.0f, y_neg = 0.0f, c_neg = 0.0f, x_pos = 0.0f, y_pos = 0.0f, c_pos = 0.0f;
        
        for(int py = y * SUBGRID_DIM; py < y * SUBGRID_DIM + SUBGRID_DIM; py++){
            for(int px = x * SUBGRID_DIM; px < x * SUBGRID_DIM + SUBGRID_DIM; px++){
                int k = py * GRID_DIM + px;
                if(ugrid_p_c[k]){
                    c_pos += ugrid_p_c[k];
                    x_pos += ugrid_p_c[k] * ugrid_p_x[k];
                    y_pos += ugrid_p_c[k] * ugrid_p_y[k];
                } 
                
                if(ugrid_n_c[k]){
                    c_neg += ugrid_n_c[k];
                    x_neg += ugrid_n_c[k] * ugrid_n_x[k];
                    y_neg += ugrid_n_c[k] * ugrid_n_y[k];
                }
            }
        }

        if(c_pos != 0){
            x_pos /= c_pos;
            y_pos /= c_pos;
            ugrid_p_x[GRID_SIZE + tid] = x_pos;
            ugrid_p_y[GRID_SIZE + tid] = y_pos;
        } 

        ugrid_p_c[GRID_SIZE + tid] = c_pos; 

        if(c_neg != 0){
            x_neg /= c_neg;
            y_neg /= c_neg;
            ugrid_n_x[GRID_SIZE + tid] = x_neg;
            ugrid_n_y[GRID_SIZE + tid] = y_neg;
        } 

        ugrid_n_c[GRID_SIZE + tid] = c_neg;         
    }

// Update simulation using cuda
void cudaUpdateTexture(ElectricFieldCuda * electricField, float dt) {    
    uchar4* d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, NULL,
        cuda_pbo_resource);

    dim3 pixel_blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 pixel_gridSize(ceilPosInt(electricField->width, pixel_blockSize.x), ceilPosInt(electricField->height, pixel_blockSize.y));

    
    dim3 particle_blockSize(MAX_THREADS);
    dim3 particle_gridSize(ceilPosInt(electricField->p_count, particle_blockSize.x));

    dim3 grid_blockSize(min(MAX_THREADS, GRID_SIZE));
    dim3 grid_gridSize(ceilPosInt(GRID_SIZE, grid_blockSize.x));

    dim3 majorGrid_blockSize(MAJOR_GRID_DIM, MAJOR_GRID_DIM);

    #ifdef MEASURE_TIME
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    cudaEventRecord(start);

    #endif

    // Calculate Cell_ID for each particle
    calculateIndices<<<particle_gridSize, particle_blockSize>>>(electricField->p_x, electricField->p_y, electricField->cIndices, electricField->pIndices,
                                                                 electricField->width, electricField->height, electricField->p_count);
    
    thrust::device_ptr<int> cIndices_ptr = thrust::device_pointer_cast(electricField->cIndices);
    thrust::device_ptr<int> pIndices_ptr = thrust::device_pointer_cast(electricField->pIndices);

    // Sort particles by Cell_ID
    thrust::sort_by_key(cIndices_ptr, cIndices_ptr+electricField->p_count, pIndices_ptr);

    thrust::device_ptr<int> cIndices_start_ptr = thrust::device_pointer_cast(electricField->cIndices_start);
    thrust::device_ptr<int> cIndices_end_ptr = thrust::device_pointer_cast(electricField->cIndices_end);

    thrust::counting_iterator<int> iter(0);

    // Calculate starting index for each cell
    thrust::lower_bound(cIndices_ptr, cIndices_ptr + electricField->p_count, iter, iter + GRID_SIZE, cIndices_start_ptr);

    // Calculate ending index for each cell
    thrust::upper_bound(cIndices_ptr, cIndices_ptr + electricField->p_count, iter, iter + GRID_SIZE, cIndices_end_ptr);
    
    // calculate virtual particles
    CalculateGridParticles<<<grid_gridSize, grid_blockSize>>>(electricField->p_x, electricField->p_y, electricField->p_c,
                                                            electricField->uniform_grid_p_x, electricField->uniform_grid_p_y, electricField->uniform_grid_p_c, 
                                                            electricField->uniform_grid_n_x, electricField->uniform_grid_n_y, electricField->uniform_grid_n_c, 
                                                            electricField->cIndices_start, electricField->cIndices_end, electricField->pIndices
                                                                            );
                                                                            
    // calculate virtual major particles
    CalculateMajorGridParticles<<<1, majorGrid_blockSize>>>(electricField->p_x, electricField->p_y, electricField->p_c,
                                                            electricField->uniform_grid_p_x, electricField->uniform_grid_p_y, electricField->uniform_grid_p_c, 
                                                            electricField->uniform_grid_n_x, electricField->uniform_grid_n_y, electricField->uniform_grid_n_c,
                                                            electricField->pIndices
                                                            );
    
    #ifdef MEASURE_TIME 
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    prepareGridTime += milliseconds;
    cudaEventRecord(start);

    #endif
    
    // Update electric field potentials
    UpdateField<<<pixel_gridSize, pixel_blockSize>>>(electricField->p_x, electricField->p_y, electricField->p_c, electricField->potential_x, 
                    electricField->potential_y, electricField->cIndices_start, electricField->cIndices_end, electricField->pIndices,
                    electricField->width, electricField->height,  d_out, electricField->uniform_grid_p_x, electricField->uniform_grid_p_y, 
                    electricField->uniform_grid_p_c, electricField->uniform_grid_n_x, electricField->uniform_grid_n_y, electricField->uniform_grid_n_c);

    #ifdef MEASURE_TIME 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    updateFieldTime += milliseconds;
    cudaEventRecord(start);

    #endif


    // Update particles positions and velocities
    UpdateParticles<<<particle_gridSize, particle_blockSize>>>(electricField->p_x, electricField->p_y, electricField->p_c, electricField->p_m,
                                                            electricField->p_Vx, electricField->p_Vy, electricField->potential_x, electricField->potential_y,
                                                            electricField->p_count, electricField->width, electricField->height, dt);

    #ifdef MEASURE_TIME 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    updateParticlesTime += milliseconds;

    #endif
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void cudaRegisterPBO() {
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
        cudaGraphicsMapFlagsWriteDiscard);
}
