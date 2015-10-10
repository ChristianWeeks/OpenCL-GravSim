
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
cl_kernel vverletKernel, emitterKernel, resetEmitKernel;
cl_program myprogram;
cl_mem oclvbo, oclcbo, dev_velocity, dev_rseed, dev_activeParticles, dev_emitSum, dev_mass, dev_gridGravity, dev_hashTable;
size_t worksize[] = {NUMBER_OF_PARTICLES}; 
size_t lws[] = {128}; 

float host_position[NUMBER_OF_PARTICLES][4];
float host_mass[NUMBER_OF_PARTICLES];
float host_velocity[NUMBER_OF_PARTICLES][4];
float host_color[NUMBER_OF_PARTICLES][4];
float host_rseed[NUMBER_OF_PARTICLES];
int host_gridHashTable[SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS][GRID_HASH_LEN];
//x y z contain direction of force, w contains magnitude
float host_gridGravity[SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS][4];
float host_emitSum[1];
int host_activeParticles[NUMBER_OF_PARTICLES] = {0};

Camera *camera;

void initParticles(){
    for(int i = 0; i < NUMBER_OF_PARTICLES; i++){
            host_rseed[i] = genrand(); 
            host_mass[i] = host_rseed[i]*2;
            if(!(i % 10000)) host_mass[i] += host_rseed[i]*100;
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
    clSetKernelArg(emitterKernel, 6, sizeof(float), &emitPos[0]);
    clSetKernelArg(emitterKernel, 7, sizeof(float), &emitPos[1]);
    clSetKernelArg(emitterKernel, 8, sizeof(float), &emitPos[2]);
    clEnqueueNDRangeKernel(mycommandqueue,emitterKernel,1,NULL,worksize,lws,0,0,&waitlist[0]);
    clWaitForEvents(1,waitlist);
    if(timeCounter > emissionSpacing + emissionLength){
        emitPos[0] = 2.0*genrand() - 1.0;
        emitPos[1] = 2.0*genrand() - 1.0;
        emitPos[2] = 2.0*genrand() - 1.0;
        timeCounter = 0;
    }
}
clEnqueueNDRangeKernel(mycommandqueue,vverletKernel,1,NULL,worksize,lws,0,0,
	&waitlist[0]);
clWaitForEvents(1,waitlist);
}

void mydisplayfunc()
{
glFinish();
clEnqueueAcquireGLObjects(mycommandqueue,1,&oclvbo,0,0,0);
do_kernel();
clEnqueueReleaseGLObjects(mycommandqueue,1, &oclvbo, 0,0,0);
clEnqueueReadBuffer(mycommandqueue,oclcbo, CL_FALSE, 0, sizeof(float)*4*NUMBER_OF_PARTICLES, host_color, 0, NULL, NULL);
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
int gridSize = SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS;


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
myprogram = clCreateProgramWithSource(mycontext,1,(const char **)&oclsource,
	&program_length, &err);
if(err==CL_SUCCESS) fprintf(stderr,"create ok\n");
else fprintf(stderr,"create err %d\n",err);
clBuildProgram(myprogram, 0, NULL, NULL, NULL, NULL);
vverletKernel = clCreateKernel(myprogram, "VVerlet", &err);
resetEmitKernel = clCreateKernel(myprogram, "resetEmitter", &err);
emitterKernel = clCreateKernel(myprogram, "emitParticles", &err);
if(err==CL_SUCCESS) fprintf(stderr,"build ok\n");
else fprintf(stderr,"build err %d\n",err);

glBindBuffer(GL_ARRAY_BUFFER, OGL_VBO);
glBufferData(GL_ARRAY_BUFFER, DATA_SIZE, &host_position[0][0], GL_DYNAMIC_DRAW);
oclvbo = clCreateFromGLBuffer(mycontext,CL_MEM_WRITE_ONLY,OGL_VBO,&err);

oclcbo= clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	DATA_SIZE,&host_color[0][0],&err); 

dev_velocity = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	DATA_SIZE,&host_velocity[0][0],&err); 

dev_gridGravity= clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	gridSize*sizeof(float)*4,&host_gridGravity[0],&err); 

dev_hashTable = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	gridSize*sizeof(int)*GRID_HASH_LEN,&host_gridHashTable[0][0],&err); 

dev_mass = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	sizeof(float)*NUMBER_OF_PARTICLES,&host_velocity[0][0],&err); 


dev_rseed = clCreateBuffer(mycontext,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	NUMBER_OF_PARTICLES*sizeof(float),&host_rseed[0],&err); 

dev_activeParticles = clCreateBuffer(mycontext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	NUMBER_OF_PARTICLES*sizeof(int),&host_activeParticles[0],&err); 
dev_emitSum= clCreateBuffer(mycontext, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,
	sizeof(int),&host_emitSum,&err); 


clSetKernelArg(vverletKernel,0,sizeof(cl_mem),(void *)&oclvbo);
clSetKernelArg(vverletKernel,1,sizeof(cl_mem),(void *)&dev_velocity);
clSetKernelArg(vverletKernel,2,sizeof(cl_mem),(void *)&dev_rseed);
clSetKernelArg(vverletKernel,3,sizeof(cl_mem),(void *)&dev_activeParticles);
clSetKernelArg(vverletKernel,4,sizeof(cl_mem),(void *)&oclcbo);

clSetKernelArg(emitterKernel,0,sizeof(cl_mem),(void *)&oclvbo);
clSetKernelArg(emitterKernel,1,sizeof(cl_mem),(void *)&dev_velocity);
clSetKernelArg(emitterKernel,2,sizeof(cl_mem),(void *)&dev_rseed);
clSetKernelArg(emitterKernel,3,sizeof(cl_mem),(void *)&dev_activeParticles);
clSetKernelArg(emitterKernel,4,sizeof(cl_mem),(void *)&dev_emitSum);
clSetKernelArg(emitterKernel,5,sizeof(cl_mem),(void *)&oclcbo);

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
clReleaseMemObject(oclvbo);
clReleaseMemObject(oclcbo);
clReleaseMemObject(dev_velocity);
clReleaseMemObject(dev_rseed);
clReleaseMemObject(dev_activeParticles);
clReleaseMemObject(dev_emitSum);
clReleaseMemObject(dev_emitSum);
clReleaseMemObject(dev_emitSum);
clReleaseContext(mycontext);
exit(0);
}

void mouseEventHandler(int button, int state, int x, int y) {
  // let the camera handle some specific mouse events (similar to maya)
  camera->HandleMouseEvent(button, state, x, y);
  glutPostRedisplay();
}

void motionEventHandler(int x, int y) {
  // let the camera handle some mouse motions if the camera is to be moved
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

