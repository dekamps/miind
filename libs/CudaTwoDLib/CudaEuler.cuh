#ifndef CUEMTESTCUH
#define CUEMTESTCUH

typedef unsigned int inttype;
typedef float fptype;

__global__ void CudaCalculateDerivative(inttype N, fptype rate, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset);
__global__ void CudaSingleTransformStep(inttype N, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset);
__global__ void CudaCalculateGridDerivative(inttype N, fptype rate, fptype stays, fptype goes, inttype offset_1, inttype offset_2, fptype* derivative, fptype* mass, inttype offset);
__global__ void EulerStep(inttype N, fptype* derivative, fptype* mass, fptype timestep);
__global__ void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, fptype* mass, unsigned int* map);
__global__ void MapReset(unsigned int n_reset, unsigned int* res_from, unsigned int* res_to, fptype* res_alpha, fptype* mass, unsigned int* map, fptype* rate);
__global__ void Remap(int N, unsigned int* i_1, unsigned int t, unsigned int *map, unsigned int* first, unsigned int* length);
__global__ void ResetFinish(inttype n_reset, inttype* res_from, fptype* mass, inttype* map);
__global__ void CudaClearDerivative(inttype N, fptype* dydt, fptype* mass);
__device__ int modulo(int a, int b);


#endif
