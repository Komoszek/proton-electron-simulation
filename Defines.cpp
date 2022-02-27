#include "Defines.hpp"

GLuint pbo = 0;
GLuint tex = 0;

int window_height = 0;
int window_width = 0;

#ifdef MEASURE_TIME

int loopCount = 0;
float prepareGridTime = 0.0f;
float updateFieldTime = 0.0f;
float updateParticlesTime = 0.0f;
float frameTime = 0.0f;
float memcpyTime = 0.0f;
float dataGenerateTime = 0.0f;

#endif
