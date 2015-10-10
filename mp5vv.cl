#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#define STEPS_PER_RENDER 10
#define MASS 1.0f
#define DELTA_T (0.002f)
#define FRICTION 0.25f
#define RESTITUTION 0.9f
#define EMIT_RATE 20000

#define EPS_DOWN (0.01f) // gravity
#define V_DRAG (3.5f)


int getGridIndex(float4 p){
    //int4 index;
    int3 iIndex;
    float3 fIndex;
    float cellLength = SPATIAL_GRID_SIZE / SPATIAL_GRID_SEGMENTS;
    //printf("%0.2f", cellLength);
    fIndex.x = (p.x + SPATIAL_GRID_SIZE/2.0) / (float)(cellLength);
    fIndex.y = (p.y + SPATIAL_GRID_SIZE/2.0) / (float)(cellLength);
    fIndex.z = (p.z + SPATIAL_GRID_SIZE/2.0) / (float)(cellLength);
    iIndex.x = (int)fIndex.x;
    iIndex.y = (int)fIndex.y;
    iIndex.z = (int)fIndex.z;
    if(iIndex.x < SPATIAL_GRID_SEGMENTS && iIndex.x >= 0 && iIndex.y < SPATIAL_GRID_SEGMENTS && iIndex.y >= 0
            && iIndex.z < SPATIAL_GRID_SEGMENTS && iIndex.z >= 0 ){
        return iIndex.x + SPATIAL_GRID_SEGMENTS*iIndex.z + SPATIAL_GRID_SEGMENTS*SPATIAL_GRID_SEGMENTS*iIndex.y;}
    //if the particles falls outside the bounds of our table, return -1
    return -1;
}

//int hashParticles(__global float4 p, __global int gridHash){



//}

float4 getforce(float4 pos, float4 vel)
{
float4 force;

force.x = (-2.0f*pos.z - pos.x) - V_DRAG*vel.x;
force.y = EPS_DOWN -V_DRAG*vel.y;
force.z = (2.0f*pos.x - pos.z) -V_DRAG*vel.z;
force.w = 1.0f;
force = force*(float)(0.6);
return(force);
}

float vecMagn(float4 vec){
    return (float)(sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z));
}

float4 getGridColor(int index){
    float4 color;
    color.w = 1;
    index = index%6;
    //modulo can yield negatives
    if(index < 0)
        index += -2*index;
    if(index == 0){
        color.x = 1;
        color.y = 0;
        color.z = 0;
    }
    else if(index == 1){
        color.x = 0;
        color.y = 1;
        color.z = 0;
    }
    else if(index == 2){
        color.x = 0;
        color.y = 0;
        color.z = 1;
    }
    else if(index == 3){
        color.x = 1;
        color.y = 0;
        color.z = 1;
    }
    else if(index == 4){
        color.x = 0;
        color.y = 1;
        color.z = 1;
    }
    else{
        color.x = 1;
        color.y = 1;
        color.z = 0;
    }
    return color;
}

float truncColor(float c){
    if(c > 1.0) c = 1.0;
    if(c < 0.0) c = 0.0; 
    return c;
}
float4 getColor(float4 v, float4 c){
    
    float magn = vecMagn(v);
    if(magn > 1.0){
        magn = 1.0;
    }
    float4 newColor;
    newColor.x = c.x; 
    if(c.y < 0.6)
        newColor.y += 0.5*DELTA_T;
    else
        newColor.y = 1.0 - magn;
    newColor.z = c.z;
    newColor.w = 1 - (1-magn)*0.4;
    return newColor;
}

float dotProd(float4 vec1, float4 vec2){
    float d = vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
   return d;
}

#define MULT (87.0f)
#define MOD (3647.0f)

float goober(float prev)
{
prev *= (MOD*MULT);
return(fmod(prev,MOD)/MOD);
}

__kernel void VVerlet(__global float4* p, __global float4* v, __global float* r, __global int* activeP, __global float4* color)
{
unsigned int i = get_global_id(0);
float4 force, zoom;
float radius, mylength;
if(activeP[i]){
    for(int steps=0;steps<STEPS_PER_RENDER;steps++){
        force = getforce(p[i],v[i]);
        v[i] += force*DELTA_T/2.0f;
        p[i] += v[i]*DELTA_T;
        force = getforce(p[i],v[i]);
        v[i] += force*DELTA_T/2.0f;
        //color[i] = getColor(v[i], color[i]); 
        color[i] = getGridColor(getGridIndex(p[i]));
        //printf("grid Index: %d\n", (int)getGridIndex(p[i]));


        radius = sqrt(p[i].x*p[i].x + p[i].z*p[i].z);
        if((radius< 0.05f)||(p[i].y<0.0f)||vecMagn(v[i]) < 0.005){
            //goober shifts slowly to 1. We semi-reset it here
            r[i] = 1 - goober(r[i]);
            //activeP[i] = 0;

            }
        else{
            //we check for sphere collisions
            float4 sphereCenters[4]; 
            sphereCenters[0].x = -0.7;
            sphereCenters[0].y = 0.0;
            sphereCenters[0].z = -0.7;
            sphereCenters[1].x = 0.7;
            sphereCenters[1].y = 0.0; 
            sphereCenters[1].z = -0.7;
            sphereCenters[2].x = -0.7;
            sphereCenters[2].y = 0.0; 
            sphereCenters[2].z = 0.7;
            sphereCenters[3].x = 0.7;
            sphereCenters[3].y = 0.0; 
            sphereCenters[3].z = 0.7;
            for(int k = 0; k < 4; k++){
                sphereCenters[k].w = 1.0;
                //printf("sphereCenter: %.2f, %.2f, %.2f, %.2f\n", sphereCenters[k].x, sphereCenters[k].y, sphereCenters[k].z, sphereCenters[k].w);
                float4 distVec =  p[i] -  sphereCenters[k];
                float dist = vecMagn(distVec);
                if(dist < 0.4){
                    float4 normalVec = distVec / dist;

                    // Bounce the point.  Usually it's 
                    // vout = vin - (1+r)(vin o n)n
                    //      - f*(vin - (vin o n)n)/||vin - (vin o n)n||
                    // but here (vin o n)n) = (v[i].x,0,0), and so
                    // vin - (vin o n)n = (0,v[i].y,v[i].z).
                    float4 vn = dot(v[i],normalVec)*normalVec;
                    float4 vp = v[i] - vn;
                    v[i] = (1-FRICTION)*vp - RESTITUTION*vn;
                    //v[i].x = v[i].x - ((1.0 + RESTITUTION)*v[i].x);
                    //shift it outside of the sphere
                    float delta = 0.4 - dist;
                    p[i] += normalVec * (delta);

                    if(k == 1){
                        color[i].z += 1.0;
                        color[i].x -= 0.6;
                    }else if(k == 2){
                        color[i].z += 0.8;
                        color[i].y -= 0.4;
                        color[i].x += 0.4;    
                    }else if(k == 3){
                        color[i].z = 0.6;
                        color[i].y = 0.1;
                        color[i].x = 0.9;
                    }
                    else{
                        color[i].z -= 0.5;
                        color[i].x -= 0.8;
                        color[i].y += 0.8;
                    }
                    color[i].x = truncColor(color[i].x);
                    color[i].y = truncColor(color[i].y);
                    color[i].z = truncColor(color[i].z);
                    break;
                    }
                }
            }
        }
    p[i].w = 1.0f;
    }
}
__kernel void resetEmitter(__global int * sum){
    if(get_local_id(0) == 0)
        sum[0] = 0;
}

__kernel void emitParticles(__global float4* p, __global float4* v, __global float* r, __global int* activeP, __global int * sum, __global float4* color, float emitX, float emitY, float emitZ){
    unsigned int i = get_global_id(0);
    unsigned int j;
    if(!activeP[i] && sum[0] < EMIT_RATE){
        atomic_inc(&sum[0]); 
        r[i] = goober(r[i]);
        activeP[i] = 1;

        float vel = 10.0;
        p[i].x = 0.3f*r[i] - 0.15;
        r[i] = goober(r[i]);
        color[i].x = 1.0;
        p[i].y = 0.3f*r[i] - 0.15;
        r[i] = goober(r[i]);
        color[i].z = 0.2 + 0.3*r[i];
        p[i].z = 0.3f*r[i] - 0.15;
        p[i].w = 1.0;
        r[i] = goober(r[i]);
        color[i].y = 0.2 + 0.3*r[i];
        v[i].x = vel*r[i] - vel / 2.0;
        r[i] = goober(r[i]);
        v[i].y = vel*r[i] - vel / 2.0;
        r[i] = goober(r[i]);
        v[i].z = vel*r[i] - vel / 2.0; 
        v[i].w = 1.0;
        color[i] = getColor(v[i], color[i]);
    }
}
