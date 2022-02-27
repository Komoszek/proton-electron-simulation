#ifndef PE_SIMULATION_CPU_HPP
#define PE_SIMULATION_CPU_HPP

#include <cmath>
#include "ElectricFieldCpu.hpp"
#include "Defines.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

void updateTextureCPU(struct ElectricField * electricField, float dt);

// Calculates 1D index of cell that position belongs to 
static int posToSubIndices(float x, float dim){
    int p = (x / dim) * GRID_DIM;
    
    return p < GRID_DIM ? p : GRID_DIM - 1;
}

// Calculates electric field potential on (x, y) created by particle 
static void CalculateEF(float p_x, float p_y, float p_c, float x, float y, float &ef_x, float &ef_y, float &ef_potential){
        float dx = x - p_x;
        float dy = y - p_y;

        float distanceInvSqrt = 1/sqrt(dx*dx + dy*dy + SOFTENING);
        
        ef_potential = K_E * p_c * distanceInvSqrt * distanceInvSqrt;

        ef_x = ef_potential * dx * distanceInvSqrt;
        ef_y = ef_potential * dy * distanceInvSqrt;
}

// Clamps value v to [min, max]
static float clamp(float v, float min, float max){
  return fminf(fmaxf(v, min), max);
}

// Calculates electric field potential on (x, y) created by all particles referenced between pStart-th and pEnd-th position in pIndices array
static void CalculateEFFromGrid(float * p_x, float * p_y, float * p_c, float x, float y, int pStart, int pEnd, 
                                    std::pair<int, int> * cpIndices, float &ef_x, float &ef_y, float &ef_potential){

    float temp_ef_x = 0.0f, temp_ef_y  = 0.0f, temp_ef_potential  = 0.0f;
    
    for(int p = pStart; p < pEnd; p++){
            int pIndex = cpIndices[p].second;
            CalculateEF(p_x[pIndex], p_y[pIndex], p_c[pIndex], x, y, temp_ef_x, temp_ef_y, temp_ef_potential); 

            ef_x += temp_ef_x;
            ef_y += temp_ef_y;
            ef_potential += temp_ef_potential;
        }
}

// Calculates and draws potential created by all particles on single pixel 
static void UpdateField(ElectricFieldCpu * ef){

    for(int y = 0; y < ef->height; y++ ){
        for(int x = 0; x < ef->width; x++){   
            int id = y * ef->width + x;     
            float pos_x = x + 0.5f;
            float pos_y = y + 0.5f;

            float ef_x = 0.0f, ef_y = 0.0f, ef_potential = 0.0f;
            float temp_ef_x = 0.0f, temp_ef_y = 0.0f, temp_ef_potential = 0.0f;

            // Calculate x, y cell indices and get neighboring cell rectangle
            int subind_x = posToSubIndices(pos_x, ef->width);
            int subind_y = posToSubIndices(pos_y, ef->height);

            int subgrid_n_minx = std::max(subind_x-1, 0);
            int subgrid_n_miny = std::max(subind_y-1, 0);

            int subgrid_n_maxx = std::min(subind_x+1, GRID_DIM-1);
            int subgrid_n_maxy = std::min(subind_y+1, GRID_DIM-1);

            int k;
            // Calculate electric field potential created by real particles from adjacent cells
            for(int j = subgrid_n_miny; j <= subgrid_n_maxy; j++ ){
                k = j * GRID_DIM + subgrid_n_minx;
                for(int i = subgrid_n_minx; i <= subgrid_n_maxx; i++, k++){
                    CalculateEFFromGrid(ef->p_x, ef->p_y, ef->p_c, pos_x, pos_y, ef->cIndices_start[k], 
                                    ef->cIndices_end[k], ef->cpIndices, ef_x, ef_y, ef_potential);
                }
            }
            

            subgrid_n_minx /= SUBGRID_DIM;
            subgrid_n_maxx /= SUBGRID_DIM;

            subgrid_n_miny /= SUBGRID_DIM;
            subgrid_n_maxy /= SUBGRID_DIM;

            /*
            // Calculate electric field potential created by virtual major particles from non-adjacent major grid cells
            k = GRID_SIZE;
            for(int i = 0; i < MAJOR_GRID_DIM; i++){
                for(int j = 0; j < MAJOR_GRID_DIM; j++, k++){
                    if(j >= subgrid_n_minx && j <= subgrid_n_maxx && i >= subgrid_n_miny && i <= subgrid_n_maxy)
                        continue;
                    
                    if(ef->uniform_grid_p_c[k]){
                        CalculateEF(ef->uniform_grid_p_x[k], ef->uniform_grid_p_y[k], ef->uniform_grid_p_c[k], 
                            pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                        ef_x += temp_ef_x;
                        ef_y += temp_ef_y;
                        ef_potential += temp_ef_potential;
                    }
                    

                    if(ef->uniform_grid_n_c[k]){
                        CalculateEF(ef->uniform_grid_n_x[k], ef->uniform_grid_n_y[k], ef->uniform_grid_n_c[k], 
                            pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                        ef_x += temp_ef_x;
                        ef_y += temp_ef_y;
                        ef_potential += temp_ef_potential;
                    }
                }
            }*/

            
            
            subgrid_n_minx *= SUBGRID_DIM;
            subgrid_n_miny *= SUBGRID_DIM;

            subgrid_n_maxx = (subgrid_n_maxx + 1) * SUBGRID_DIM;
            subgrid_n_maxy = (subgrid_n_maxy + 1) * SUBGRID_DIM;
            
            // Calculate electric field potential created by virtual particles from non-adjacent cells but in adjacent major cell
            for(int j = subgrid_n_miny; j < subgrid_n_maxy; j++){
                k = j * GRID_DIM + subgrid_n_minx;
                for(int i = subgrid_n_minx; i < subgrid_n_maxx; i++, k++){ 
                    int d = std::max(std::abs(i -  subind_x), std::abs(j -  subind_y));
                    if(d < 2)
                        continue;

                    if(ef->uniform_grid_p_c[k]){
                        CalculateEF(ef->uniform_grid_p_x[k], ef->uniform_grid_p_y[k], ef->uniform_grid_p_c[k], 
                            pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                        ef_x += temp_ef_x;
                        ef_y += temp_ef_y;
                        ef_potential += temp_ef_potential;
                    }
                    

                    if(ef->uniform_grid_n_c[k]){
                        CalculateEF(ef->uniform_grid_n_x[k], ef->uniform_grid_n_y[k], ef->uniform_grid_n_c[k], 
                            pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential);
                        ef_x += temp_ef_x;
                        ef_y += temp_ef_y;
                        ef_potential += temp_ef_potential;
                    }
                }
            }
            
            // save x and y potentials for later calculations
            ef->potential_x[id] = ef_x;
            ef->potential_y[id] = ef_y;

            // draw potential in pixel

            ef_potential *= DRAW_COEF;

            unsigned char red = 0;
            unsigned char blue = 0;

            if(ef_potential >= 0.0f){
                red = clamp(ef_potential, 0.0f, 255.0f);
            } else {
                blue  = clamp(-ef_potential, 0.0f, 255.0f);
            }
            int p = id * 4;

            ef->pixels[p]  = red;
            ef->pixels[p + 1]  = 0;
            ef->pixels[p + 2]  = blue;
            ef->pixels[p + 3]  = 255;
        }
    }
}

// Updates particles speeds and positions using electric field potential
static void UpdateParticles(ElectricFieldCpu * ef, float dt){
    float temp_ef_x, temp_ef_y, temp_ef_potential;

    for(int i = 0; i < ef->p_count; i++){
        // Get nearest pixel indices
        int g_y = (int)ef->p_y[i];
        int g_x = (int)ef->p_x[i];
        
        if(g_x >= ef->width){
            g_x = ef->width -1;
        }

        if(g_y >= ef->height){
            g_y = ef->height -1;
        }

        // get potentials of pixel
        float ef_x = ef->potential_x[g_y * ef->width + g_x];
        float ef_y = ef->potential_y[g_y * ef->width + g_x];

        // translate grid indices to position
        float pos_x = g_x + 0.5f;
        float pos_y = g_y + 0.5f;

        // calculate electric field created by particle
        CalculateEF(ef->p_x[i], ef->p_y[i], ef->p_c[i], pos_x, pos_y, temp_ef_x, temp_ef_y, temp_ef_potential); 
        
        // adjust potential by removing potential created by particle
        ef_x -= temp_ef_x;
        ef_y -= temp_ef_y;

        float pd = ef->p_c[i] / ef->p_m[i];

        // a = F / m, F = q * E => a = q * E / m
        
        float ax = ef_x * pd;
        float ay = ef_y * pd;
        
        
        ef->p_Vx[i] += ax * dt;
        ef->p_Vy[i] += ay * dt;
        
        ef->p_x[i] += ef->p_Vx[i] * dt;
        ef->p_y[i] += ef->p_Vy[i] * dt;

        if(ef->p_x[i] <= 0.0f){
            ef->p_x[i] = BORDER_EPS;
            ef->p_Vx[i] = -ef->p_Vx[i] * VELOCITY_DRAG;
        } else if(ef->p_x[i] >= ef->width){
            ef->p_x[i] = ef->width - BORDER_EPS;
            ef->p_Vx[i] = -ef->p_Vx[i] * VELOCITY_DRAG;
        }

        if(ef->p_y[i] <= 0.0f){
            ef->p_y[i] = BORDER_EPS;
            ef->p_Vy[i] = -ef->p_Vy[i] * VELOCITY_DRAG;
        } else if(ef->p_y[i] >= ef->height){
            ef->p_y[i] = ef->height - BORDER_EPS;
            ef->p_Vy[i] = -ef->p_Vy[i] * VELOCITY_DRAG;
        }
    }

}

// Calculates cell indices for particles
static void calculateIndices(ElectricFieldCpu * ef){
    for(int i = 0; i < ef->p_count; i++){
        int x = posToSubIndices(ef->p_x[i], ef->width);
        int y = posToSubIndices(ef->p_y[i], ef->height);

        if(x >= GRID_DIM) x = GRID_DIM - 1;
        if(y >= GRID_DIM) y = GRID_DIM - 1;

        ef->cpIndices[i].first = y * GRID_DIM + x;
        ef->cpIndices[i].second = i;
    }
}


// Calculates virtual particles placed inside grid cells
static void CalculateGridParticles(ElectricFieldCpu * ef){

        for(int i = 0; i < GRID_SIZE; i++){
            float x_neg = 0.0f, y_neg = 0.0f, c_neg = 0.0f, x_pos = 0.0f, y_pos = 0.0f, c_pos = 0.0f;
                
            for(int p = ef->cIndices_start[i]; p < ef->cIndices_end[i]; p++){
                int pIdx = ef->cpIndices[p].second;

                if(ef->p_c[pIdx] > 0.0f){
                    c_pos += ef->p_c[pIdx];
                    x_pos += ef->p_c[pIdx] * ef->p_x[pIdx];
                    y_pos += ef->p_c[pIdx] * ef->p_y[pIdx];
                } else if(ef->p_c[pIdx] < 0.0f){
                    c_neg += ef->p_c[pIdx];
                    x_neg += ef->p_c[pIdx] * ef->p_x[pIdx];
                    y_neg += ef->p_c[pIdx] * ef->p_y[pIdx];
                }
            }

            if(c_pos != 0){
                x_pos /= c_pos;
                y_pos /= c_pos;
                ef->uniform_grid_p_x[i] = x_pos;
                ef->uniform_grid_p_y[i] = y_pos;
            } 

            ef->uniform_grid_p_c[i] = c_pos; 

            if(c_neg != 0){
                x_neg /= c_neg;
                y_neg /= c_neg;
                ef->uniform_grid_n_x[i] = x_neg;
                ef->uniform_grid_n_y[i] = y_neg;
            } 

            ef->uniform_grid_n_c[i] = c_neg;  
        }       
}

// Calculates virtual particles placed inside Major Grid Cells
static void CalculateMajorGridParticles(ElectricFieldCpu * ef){
    for(int j = 0; j < MAJOR_GRID_DIM; j++){
        for(int i = 0; i < MAJOR_GRID_DIM; i++){
            int id = GRID_SIZE + j * MAJOR_GRID_DIM + i;
            float x_neg = 0.0f, y_neg = 0.0f, c_neg = 0.0f, x_pos = 0.0f, y_pos = 0.0f, c_pos = 0.0f;
            
            for(int py = j * SUBGRID_DIM; py < j * SUBGRID_DIM + SUBGRID_DIM; py++){
                for(int px = i * SUBGRID_DIM; px < i * SUBGRID_DIM + SUBGRID_DIM; px++){
                    int k = py * GRID_DIM + px;
                    if(ef->uniform_grid_p_c[k]){
                        c_pos += ef->uniform_grid_p_c[k];
                        x_pos += ef->uniform_grid_p_c[k] * ef->uniform_grid_p_x[k];
                        y_pos += ef->uniform_grid_p_c[k] * ef->uniform_grid_p_y[k];
                    } 
                    
                    if(ef->uniform_grid_n_c[k]){
                        c_neg += ef->uniform_grid_n_c[k];
                        x_neg += ef->uniform_grid_n_c[k] * ef->uniform_grid_n_x[k];
                        y_neg += ef->uniform_grid_n_c[k] * ef->uniform_grid_n_y[k];
                    }
                }
            }

            if(c_pos != 0){
                x_pos /= c_pos;
                y_pos /= c_pos;
                ef->uniform_grid_p_x[id] = x_pos;
                ef->uniform_grid_p_y[id] = y_pos;
            } 

            ef->uniform_grid_p_c[id] = c_pos; 

            if(c_neg != 0){
                x_neg /= c_neg;
                y_neg /= c_neg;
                ef->uniform_grid_n_x[id] = x_neg;
                ef->uniform_grid_n_y[id] = y_neg;
            } 

            ef->uniform_grid_n_c[id] = c_neg; 
        }
    }
}

static void CalculateLowerBounds(std::pair<int, int> * cpIndices, int * cIndices_start, int p_count){
    int s = 0;
    for(int i = 0; i < GRID_SIZE; i++){
        while(s < p_count && i > cpIndices[s].first){
            s++;
        }
        cIndices_start[i] = s;
    }
}

static void CalculateUpperBounds(std::pair<int, int> * cpIndices, int * cIndices_end, int p_count){
    int s = 0;
    for(int i = 0; i < GRID_SIZE; i++){
        while(s < p_count && i >= cpIndices[s].first){
            s++;
        }
        cIndices_end[i] = s;
    }
}

// Update simulation using cuda
void UpdateTextureCPU(ElectricFieldCpu * electricField, float dt) {   
    #ifdef MEASURE_TIME

    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> stop;

    float milliseconds;

    #endif

    electricField->pixels = (unsigned char *)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);


    #ifdef MEASURE_TIME

    start = std::chrono::high_resolution_clock::now();

    #endif

    // Calculate Cell ID for each particle
    calculateIndices(electricField);

    // Sort particles ids by Cell ID
    std::sort(electricField->cpIndices, electricField->cpIndices + electricField->p_count, 
            [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                return a.first < b.first;
    });

    // calculates start and ends of perticles in cells
    CalculateLowerBounds(electricField->cpIndices, electricField->cIndices_start, electricField->p_count);
    CalculateUpperBounds(electricField->cpIndices, electricField->cIndices_end, electricField->p_count);

    // calculate virtual particles
    CalculateGridParticles(electricField);

    // calculate virtual major particles
    CalculateMajorGridParticles(electricField);

    #ifdef MEASURE_TIME

    stop = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
    prepareGridTime += milliseconds;
    start = std::chrono::high_resolution_clock::now();

    #endif

    // update field potentials
    UpdateField(electricField);

    #ifdef MEASURE_TIME

    stop = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
    updateFieldTime += milliseconds;
    start = std::chrono::high_resolution_clock::now();

    #endif

    // update particles positions/velocities
    UpdateParticles(electricField, dt);

    #ifdef MEASURE_TIME 
    
    stop = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()  / 1000.f;
    updateParticlesTime += milliseconds;

    #endif

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
}

#endif