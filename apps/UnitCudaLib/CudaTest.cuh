#ifndef CUEMTESTCUH
#define CUEMTESTCUH

typedef float fptype;

__global__ void CalculateDerivative(unsigned int N, fptype* derivative, fptype* mass, fptype* val, unsigned int* ia, unsigned int* ja, unsigned int* map);
__global__ void EulerStep(unsigned int N, unsigned int n_div, fptype* derivative, fptype* mass, fptype timestep, fptype rate);
__global__ void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, fptype* mass, unsigned int* map);
__global__ void MapReset(unsigned int n_reset, unsigned int* res_from, unsigned int* res_to, fptype* res_alpha, fptype* mass, unsigned int* map, float* rate);
__global__ void Remap(int N, unsigned int* i_1, unsigned int t, unsigned int *map, unsigned int* first, unsigned int* length);
__global__ void MapFinish(unsigned int n_reset, unsigned int* res_from, fptype* mass, unsigned int* map);

__device__ int modulo(int a, int b);


#endif
