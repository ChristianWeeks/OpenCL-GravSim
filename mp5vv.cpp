// 
// This particle sytem with collisions uses velocity Verlet integration.
// mp5vv.cpp

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <unistd.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glx.h>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_ext.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_platform.h>
#include <CL/opencl.h>
#include <vector>
#include "Camera.h"
#include "gravity.h"
#include "RGU.h"

#define AXES 1
#define BOX 2
#define WALL 3
#define BORDER 4
#define GRIDBOUNDS 5
#define WIDTH 1920
#define HEIGHT 1080

GLuint OGL_VBO = 1;
GLuint OGL_CBO = 2;
#define NUMBER_OF_PARTICLES 1024*1024 
#define DATA_SIZE (NUMBER_OF_PARTICLES*4*sizeof(float)) 
double lastTime = 0;
int gridSize = (SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS);
//emit every 3 seconds
double emissionSpacing = 1000.0;
double timeCounter = emissionSpacing;
double emissionTimer = 0.0;
double emissionLength = 5000.0;
float emitPos[3] = {0};

double genrand()
{
return(((double)(random()+1))/2147483649.);
}

cl_context mycontext;
cl_command_queue mycommandqueue;
cl_program myprogram;

cl_kernel vverletKernel, 
          emitterKernel, 
          resetEmitKernel,
          resetCountersKernel,
          hashParticlesKernel;
cl_mem oclvbo,
       dev_color,
       dev_velocity,
       dev_rseed,
       //bool list used to tell which particles are active
       dev_activeP,
       //Sum used to keep track of particle emission
       dev_emitSum,
       //particle mass
       dev_pMass,
       //contains grid cell center of gravity 
       dev_gridCog,
       //contains grid cell mass
       dev_gridMass,
       //list of particle indices in each grid cell
       dev_hashTable,
       //count of how many particles in each cell
       dev_gridCounter;

std::vector<cl_mem> dev_variables;
size_t worksize[] = {NUMBER_OF_PARTICLES}; 
size_t lws[] = {128}; 
size_t resetCounters_globalWorkSize[] = {8*8*8};

float host_position[NUMBER_OF_PARTICLES][4];
float host_mass[NUMBER_OF_PARTICLES];
float host_velocity[NUMBER_OF_PARTICLES][4];
float host_color[NUMBER_OF_PARTICLES][4];
float host_rseed[NUMBER_OF_PARTICLES];
int host_hashTable[8*8*8][GRID_HASH_LEN] = {0};
//x y z contain direction of force, w contains magnitude
float host_emitSum[1];
int host_activeP[NUMBER_OF_PARTICLES] = {0};

Camera *camera;

void initParticles(){
    for(int i = 0; i < NUMBER_OF_PARTICLES; i++){
            host_rseed[i] = genrand(); 
            host_mass[i] = host_rseed[i]*2;
            if(!(i % 1000)) host_mass[i] += host_rseed[i]*100;
            for(int j = 0; j < 4; j++)host_color[i][j] = 1.0;
    }
}
void do_kernel()
{
cl_event waitlist[1];

double timeSinceStart = glutGet(GLUT_ELAPSED_TIME);
double deltaTime = timeSinceStart - lastTime;
timeCounter += deltaTime;
lastTime = timeSinceStart;
if(timeCounter > emissionSpacing){
    clEnqueueNDRangeKernel(mycommandqueue,resetEmitKernel,1,NULL,lws,lws,0,0,&waitlist[0]);
    clWaitForEvents(1,waitlist);
    clSetKernelArg(emitterKernel,6, sizeof(float), &emitPos[0]);
    clSetKernelArg(emitterKernel,7, sizeof(float), &emitPos[1]);
    clSetKernelArg(emitterKernel,8, sizeof(float), &emitPos[2]);
    clEnqueueNDRangeKernel(mycommandqueue,emitterKernel,1,NULL,worksize,lws,0,0,&waitlist[0]);
    clWaitForEvents(1,waitlist);
    if(timeCounter > emissionSpacing + emissionLength){
        emitPos[0] = 2.0*genrand() - 1.0;
        emitPos[1] = 2.0*genrand() - 1.0;
        emitPos[2] = 2.0*genrand() - 1.0;
        timeCounter = 0;
    }
}
//reset our gridCounter
clEnqueueNDRangeKernel(mycommandqueue,resetCountersKernel,1,NULL,resetCounters_globalWorkSize,lws,0,0,&waitlist[0]);
//hash our particles
clWaitForEvents(1,waitlist);
clEnqueueNDRangeKernel(mycommandqueue,hashParticlesKernel,1,NULL,resetCounters_globalWorkSize,lws,0,0,&waitlist[0]);
clWaitForEvents(1,waitlist);
//run our simulation
clEnqueueNDRangeKernel(mycommandqueue,vverletKernel,1,NULL,worksize,lws,0,0,&waitlist[0]);
clWaitForEvents(1,waitlist);
}

void mydisplayfunc()
{
glFinish();
clEnqueueAcquireGLObjects(mycommandqueue,1,&oclvbo,0,0,0);
do_kernel();
clEnqueueReleaseGLObjects(mycommandqueue,1, &oclvbo, 0,0,0);
clEnqueueReadBuffer(mycommandqueue,dev_color, CL_FALSE, 0, sizeof(float)*4*NUMBER_OF_PARTICLES, host_color, 0, NULL, NULL);
clFinish(mycommandqueue);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
camera->PerspectiveDisplay(WIDTH, HEIGHT);
glMatrixMode(GL_MODELVIEW);
glEnable(GL_DEPTH_TEST);

// Frame is drawn to blend with background.
glEnable(GL_BLEND);

glBindBuffer(GL_ARRAY_BUFFER,0);
glColorPointer(4,GL_FLOAT,0,host_color);
glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
glVertexPointer(4,GL_FLOAT,0,0);

glEnableClientState(GL_VERTEX_ARRAY);
glEnableClientState(GL_COLOR_ARRAY);
glDrawArrays(GL_POINTS, 0, NUMBER_OF_PARTICLES);
glDisableClientState(GL_VERTEX_ARRAY);
glDisableClientState(GL_COLOR_ARRAY);

glCallList(AXES);
glCallList(GRIDBOUNDS);
glutSwapBuffers();
glutPostRedisplay();
}

void setup_the_viewvol()
{
float eye[] = {2.5, 2.1, 2.0};
float view[] = {0.0, 0.4, 0.0};
float up[] = {0.0, 1.0, 0.0};
Vector3d eye2(2.5, 2.1, 2.0);
Vector3d view2(0.0, 0.4, 0.0);
Vector3d up2(0.0, 1.0, 0.0);

glMatrixMode(GL_PROJECTION);
glLoadIdentity();
gluPerspective(45.0,1.78,0.1,20.0);

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();
gluLookAt(eye[0],eye[1],eye[2],view[0],view[1],view[2],up[0],up[1],up[2]);
camera = new Camera(eye2, view2, up2);

}

void build_call_lists()
{

glNewList(AXES,GL_COMPILE);
        glBegin(GL_LINES);
                glColor4f(0.0,1.0,0.0,0.8);
                glVertex3f(0.0,0.0,0.0);
                glVertex3f(0.0,1.0,0.0);
                glColor4f(1.0,0.0,0.0,0.8);
                glVertex3f(0.0,0.0,0.0);
                glVertex3f(1.0,0.0,0.0);
                glColor4f(0.0,0.0,1.0,0.8);
                glVertex3f(0.0,0.0,0.0);
                glVertex3f(0.0,0.0,1.0);
        glEnd();
glEndList();
float rad = SPATIAL_GRID_SIZE / 2.0;
glNewList(GRIDBOUNDS,GL_COMPILE);
    glBegin(GL_LINE_STRIP);
        glColor4f(1.0,0.2,0.2,0.8);
        glVertex3f(-rad,rad,-rad);
        glVertex3f(rad,rad,-rad);
        glVertex3f(rad,rad,rad);
        glVertex3f(-rad,rad,rad);
        glVertex3f(-rad,rad,-rad);

        glVertex3f(-rad,-rad,-rad);
        glVertex3f(rad,-rad,-rad);
        glVertex3f(rad,-rad,rad);
        glVertex3f(-rad,-rad,rad);
        glVertex3f(-rad,-rad,-rad);

    glEnd();
    glBegin(GL_LINES);
        glVertex3f(rad,rad,-rad);
        glVertex3f(rad,-rad,-rad);
        glVertex3f(rad,rad,rad);
        glVertex3f(rad,-rad,rad);
        glVertex3f(-rad,rad,rad);
        glVertex3f(-rad,-rad,rad);
    glEnd();
    
glEndList();
}

void InitGL(int argc, char** argv)
{
glutInit(&argc,argv);
glutInitDisplayMode(GLUT_RGBA|GLUT_DEPTH|GLUT_DOUBLE);
glutInitWindowSize(WIDTH,HEIGHT);
glutInitWindowPosition(100,50);
glutCreateWindow("my_cool_cube");

//setup_the_viewvol();

Vector3d eye2(2.5, 2.1, 2.0);
Vector3d view2(0.0, 0.4, 0.0);
Vector3d up2(0.0, 1.0, 0.0);
camera = new Camera(eye2, view2, up2);

glPointSize(1.0);
glLineWidth(4.0);
glEnable(GL_LINE_SMOOTH);
glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
glEnable(GL_DEPTH_TEST);
glClearColor(0.1,0.1,0.2,1.0);
build_call_lists();
glewInit();
return;
}

void InitCL()
{
cl_platform_id myplatform;
cl_device_id *mydevice;
cl_int err;
char* oclsource; 
char* gravityHeader;
size_t program_length;
unsigned int gpudevcount;
int gridSize = 8*8*8;

err = RGUGetPlatformID(&myplatform);
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);

mydevice = new cl_device_id[gpudevcount];
err = clGetDeviceIDs(myplatform,CL_DEVICE_TYPE_GPU,gpudevcount,mydevice,NULL);

// You need all these to get full interoperability with OpenGL:
cl_context_properties props[] = {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)myplatform,
        0};

mycontext = clCreateContext(props,1,&mydevice[0],NULL,NULL,&err);
mycommandqueue = clCreateCommandQueue(mycontext,mydevice[0],0,&err);

gravityHeader= RGULoadProgSource("gravity.h", "", &program_length);
oclsource = RGULoadProgSource("mp5vv.cl", gravityHeader, &program_length);
myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,&program_length, &err);

if(err==CL_SUCCESS) std::cout << "create ok\n";
else std::cerr << "create err" << err << "\n";

clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
vverletKernel =     clCreateKernel(myprogram, "VVerlet", &err);
resetEmitKernel =   clCreateKernel(myprogram, "resetEmitter", &err);
emitterKernel =     clCreateKernel(myprogram, "emitParticles", &err);
resetCountersKernel=clCreateKernel(myprogram, "resetCounters", &err);
hashParticlesKernel=clCreateKernel(myprogram, "hashParticles", &err);

if(err==CL_SUCCESS) std::cout << "build ok\n";
else std::cerr << "build err: " << err << "\n";

glBindBuffer(GL_ARRAY_BUFFER, OGL_VBO);
glBufferData(GL_ARRAY_BUFFER, DATA_SIZE, &host_position[0][0], GL_DYNAMIC_DRAW);

oclvbo =        clCreateFromGLBuffer(mycontext,CL_MEM_WRITE_ONLY,OGL_VBO,&err);
dev_color =     clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,DATA_SIZE,&host_color[0][0],&err); 
dev_velocity =  clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,DATA_SIZE,&host_velocity[0][0],&err); 
dev_pMass =     clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*NUMBER_OF_PARTICLES,&host_mass[0],&err); 
dev_rseed =     clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_float)*NUMBER_OF_PARTICLES,&host_rseed[0],&err); 
dev_activeP =   clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_int)*NUMBER_OF_PARTICLES,&host_activeP[0],&err); 
dev_emitSum =   clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(cl_int),&host_emitSum,&err); 
//Do not need to copy host variable
dev_hashTable = clCreateBuffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_int)*gridSize*GRID_HASH_LEN,NULL,&err); 
dev_gridCounter=clCreateBuffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_int)*gridSize,NULL,&err); 
//We must use atomic operations to add up the center of gravity.  Unfortunately, openCL does not support atomic operations on floating
//point arithmetic, so we must approximate using unsigned long ints instead.
dev_gridCog  =  clCreateBuffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_long)*gridSize*4,NULL,&err); 
dev_gridMass =  clCreateBuffer(mycontext,CL_MEM_READ_WRITE,sizeof(cl_ulong)*gridSize, NULL, &err); 

dev_variables.push_back(oclvbo);
dev_variables.push_back(dev_color);
dev_variables.push_back(dev_velocity);
dev_variables.push_back(dev_pMass);
dev_variables.push_back(dev_rseed);
dev_variables.push_back(dev_activeP);
dev_variables.push_back(dev_emitSum);

dev_variables.push_back(dev_gridCog);
dev_variables.push_back(dev_hashTable);
dev_variables.push_back(dev_gridMass);

clSetKernelArg(vverletKernel,0,sizeof(cl_mem),(void *)&oclvbo);
clSetKernelArg(vverletKernel,1,sizeof(cl_mem),(void *)&dev_velocity);
clSetKernelArg(vverletKernel,2,sizeof(cl_mem),(void *)&dev_rseed);
clSetKernelArg(vverletKernel,3,sizeof(cl_mem),(void *)&dev_activeP);
clSetKernelArg(vverletKernel,4,sizeof(cl_mem),(void *)&dev_color);
clSetKernelArg(vverletKernel,5,sizeof(cl_mem),(void *)&dev_hashTable);

clSetKernelArg(emitterKernel,0,sizeof(cl_mem),(void *)&oclvbo);
clSetKernelArg(emitterKernel,1,sizeof(cl_mem),(void *)&dev_velocity);
clSetKernelArg(emitterKernel,2,sizeof(cl_mem),(void *)&dev_rseed);
clSetKernelArg(emitterKernel,3,sizeof(cl_mem),(void *)&dev_activeP);
clSetKernelArg(emitterKernel,4,sizeof(cl_mem),(void *)&dev_emitSum);
clSetKernelArg(emitterKernel,5,sizeof(cl_mem),(void *)&dev_color);

clSetKernelArg(resetCountersKernel,0,sizeof(cl_mem),(void *)&dev_gridCounter);
clSetKernelArg(resetCountersKernel,1,sizeof(cl_mem),(void *)&dev_gridCog);
clSetKernelArg(resetCountersKernel,2,sizeof(cl_mem),(void *)&dev_gridMass);

clSetKernelArg(hashParticlesKernel,0,sizeof(cl_mem),(void *)&oclvbo);
clSetKernelArg(hashParticlesKernel,1,sizeof(cl_mem),(void *)&dev_pMass);
clSetKernelArg(hashParticlesKernel,2,sizeof(cl_mem),(void *)&dev_hashTable);
clSetKernelArg(hashParticlesKernel,3,sizeof(cl_mem),(void *)&dev_gridCounter);
clSetKernelArg(hashParticlesKernel,4,sizeof(cl_mem),(void *)&dev_gridCog);
clSetKernelArg(hashParticlesKernel,5,sizeof(cl_mem),(void *)&dev_gridMass);

clSetKernelArg(resetEmitKernel,0,sizeof(cl_mem),(void *)&dev_emitSum);
}

void cleanup()
{
clReleaseKernel(vverletKernel);
clReleaseKernel(emitterKernel);
clReleaseKernel(resetEmitKernel);
clReleaseProgram(myprogram);
clReleaseCommandQueue(mycommandqueue);
glBindBuffer(GL_ARRAY_BUFFER,OGL_VBO);
glDeleteBuffers(1,&OGL_VBO);

cl_int err;
for(unsigned int i = 0; i < dev_variables.size(); i++){
    err = clReleaseMemObject(dev_variables[i]);
    if(err == CL_INVALID_MEM_OBJECT)
        std::cerr << "Could not release CL object " << i << "\n"; 
    else
        std::cout << "Successfully released CL object " << i << "\n";
}
clReleaseContext(mycontext);
exit(0);
}

void mouseEventHandler(int button, int state, int x, int y) {
  camera->HandleMouseEvent(button, state, x, y);
  glutPostRedisplay();
}

void motionEventHandler(int x, int y) {
  camera->HandleMouseMotion(x, y);
  glutPostRedisplay();
}
void getout(unsigned char key, int x, int y)
{
switch(key) {
        case 'q':
            cleanup();
            break;
        case 'r': case 'R':
            camera->Reset();
            break;
        case 'f': case 'F':
            camera->SetCenterOfFocus(Vector3d(0,0,0));
            break;
        default:
            break;
    }
}

int main(int argc,char **argv)
{
srandom(123456789);
initParticles();
InitGL(argc, argv); 
InitCL(); 
glutMouseFunc(mouseEventHandler);
glutMotionFunc(motionEventHandler);
glutDisplayFunc(mydisplayfunc);
glutKeyboardFunc(getout);
glutMainLoop();
}
