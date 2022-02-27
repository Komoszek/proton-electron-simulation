#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <chrono>
#include <cstring>

#include "Defines.hpp"
#include "ElectricFieldCpu.hpp"
#include "ElectricFieldCuda.hpp"
#include <unistd.h>

//FPS measurement variables
int counter = 0;
int frameCount = 0;
std::chrono::time_point<std::chrono::system_clock> initialTime;
std::chrono::time_point<std::chrono::system_clock> finalTime;

ElectricField * electricField;

bool isSimulationRunning = true;

std::chrono::time_point<std::chrono::system_clock> lastFrameEndTime;
std::chrono::time_point<std::chrono::system_clock> currentFrameStartTime;

void calculateFPS() {
    frameCount++;

    finalTime = std::chrono::high_resolution_clock::now();

    int dt = std::chrono::duration_cast<std::chrono::milliseconds>(finalTime - initialTime).count();
    if (dt >= 1000)
    {
        if(isSimulationRunning){
            char title[256];
            int fps = (int)(frameCount * 1000.0f / dt);
            sprintf(title, "%s, FPS: %d", WINDOW_TITLE, fps);
            glutSetWindowTitle(title);
        }
        
        frameCount = 0;
        initialTime = finalTime;
    }
}

void display()
{   
    currentFrameStartTime = std::chrono::high_resolution_clock::now();

    if(isSimulationRunning){
        float dt = std::chrono::duration_cast<std::chrono::milliseconds>(currentFrameStartTime - lastFrameEndTime).count() / 1000.0f;
        electricField->updateTexture(dt);
    }

    #ifdef MEASURE_TIME

    std::chrono::time_point<std::chrono::system_clock> curr = std::chrono::high_resolution_clock::now();
    float milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(curr - currentFrameStartTime).count()  / 1000.f;
    frameTime += milliseconds;
    loopCount++;

    #endif

    lastFrameEndTime = currentFrameStartTime;
    
    glEnable(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, window_width, window_height, 0, GL_RGBA,
        GL_UNSIGNED_BYTE, 0);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glutSwapBuffers();


    calculateFPS();
    glutPostRedisplay();
}

void initGLUT(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow(WINDOW_TITLE);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glewInit();
}

void processKeyboard(unsigned char key, int x, int y) {

	if (key == ' '){
        char title[256];

        if(isSimulationRunning){
            sprintf(title, "%s, Stopped", WINDOW_TITLE);
        } else {
            sprintf(title, "%s, FPS: Calculating...", WINDOW_TITLE);
        }

        glutSetWindowTitle(title);

        isSimulationRunning = !isSimulationRunning;
    }
}


void usage(char * name){
    std::cout << "USAGE: " << name << "-p n [-h k -w l -c]" << std::endl;
    std::cout << "-p: particle count " << std::endl;
    std::cout << "-h: window height " << std::endl;
    std::cout << "-w: window width " << std::endl;
    std::cout << "-c: cpu-only mode " << std::endl;

    exit(EXIT_FAILURE);
}

void parseArgs(int argc, char **argv, int &p_count, bool &cpuRender){
    char c;

    while ((c = getopt (argc, argv, "p:h:w:c")) != -1)
		switch (c){
			case 'c':
				cpuRender = true;
				break;
			case 'p':
				p_count = atoi(optarg);
                break;
			case 'h':
                window_height = atoi(optarg);
                break;
			case 'w':
                window_width = atoi(optarg);
                break;
			case '?':
			default: usage(argv[0]);
		}
	if(argc>optind) usage(argv[0]);

    if(p_count == 0)
        p_count = DEFAULT_PARTICLE_COUNT;
    if(window_height == 0)
        window_height = DEFAULT_WINDOW_HEIGHT;
    if(window_width == 0)
        window_width = DEFAULT_WINDOW_WIDTH;
}

void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * window_width * window_height * sizeof(GLubyte), 0,
        GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

int main(int argc, char ** argv){
    int p_count;
    bool cpuRender;

    parseArgs(argc, argv, p_count, cpuRender);

    initGLUT(&argc, argv);
    glutDisplayFunc(display);
    glutKeyboardFunc(processKeyboard);
    initPixelBuffer();

    if(cpuRender){
        electricField = new ElectricFieldCpu(p_count, window_width, window_height);
    } else {
        electricField = new ElectricFieldCuda(p_count, window_width, window_height);
    }

    std::cout << "To stop/resume the simulation, press Spacebar" << std::endl;
    initialTime = std::chrono::high_resolution_clock::now();
    lastFrameEndTime = initialTime;
    glutMainLoop();

    delete electricField;
    
    #ifdef MEASURE_TIME
    if(loopCount > 0){
        std::cout << "Simulation mode: "<< (cpuRender ? "CPU" : "GPU") << std::endl;

        std::cout << "Number of particles: "<< p_count << std::endl;
        std::cout << "Resolution: " << window_width << "x" << window_height << std::endl; 
        std::cout << "Average frame draw time: " << frameTime / loopCount << "ms" << std::endl;
        std::cout << "Average prepareGridTime time: " << prepareGridTime / loopCount << "ms" << std::endl;
        std::cout << "Average updateField time: " << updateFieldTime / loopCount << "ms" << std::endl;
        std::cout << "Average updateParticles time: " << updateParticlesTime / loopCount << "ms" << std::endl;
        std::cout << "Data generation time: " <<  dataGenerateTime << "ms" << std::endl;
        if(!cpuRender)
            std::cout << "cudaMemcpy time: " << memcpyTime << "ms" << std::endl;
    }
    #endif

    return EXIT_SUCCESS;
}
