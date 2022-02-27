#ifndef DEFINES_HPP
#define DEFINES_HPP

#include <GL/glew.h>
#include <GL/freeglut.h>

#define MEASURE_TIME

#ifdef MEASURE_TIME

extern int loopCount;
extern float prepareGridTime;
extern float updateFieldTime;
extern float updateParticlesTime;
extern float frameTime;
extern float memcpyTime;
extern float dataGenerateTime;

#endif


extern int window_height;
extern int window_width;

extern GLuint pbo;
extern GLuint tex;

#define DEFAULT_WINDOW_HEIGHT 1024
#define DEFAULT_WINDOW_WIDTH 1024

#define SOFTENING 1e-40f

#define VELOCITY_DRAG 0.8f

#define WINDOW_TITLE "Proton/Electron Simulation"

#define PARTICLE_CONST 340.0f
#define K_E 512.0f
#define DRAW_COEF PARTICLE_CONST / K_E
#define DEFAULT_PARTICLE_COUNT 100

#define MAX_THREADS 1024
#define BLOCK_SIZE 32

#define RANDOM_SEED 2137

#define ELECTRON_MASS 1.0f
#define PROTON_MASS 1836.0f * ELECTRON_MASS

#define PROTON_CHARGE 1.0f
#define ELECTRON_CHARGE -PROTON_CHARGE

#define BORDER_EPS 1e-40

#define MAJOR_GRID_DIM 6

#define SUBGRID_DIM 5
#define GRID_DIM MAJOR_GRID_DIM * SUBGRID_DIM

#define MAJOR_GRID_SIZE MAJOR_GRID_DIM * MAJOR_GRID_DIM

#define GRID_SIZE GRID_DIM * GRID_DIM

#endif
