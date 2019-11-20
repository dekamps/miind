#ifndef CUEMTESTCUH
#define CUEMTESTCUH

typedef unsigned int inttype;
typedef float fptype;

__global__ void CudaCalculateDerivative(inttype N, fptype rate, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset);
__global__ void CudaSingleTransformStep(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset);
__global__ void EulerStep(inttype N, fptype* derivative, fptype* mass, fptype timestep);
__global__ void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, fptype* mass, unsigned int* map);
__global__ void MapResetToRefractory(unsigned int n_reset, unsigned int* res_from, fptype* mass, unsigned int* map, fptype* refactory_mass);
__global__ void MapResetShiftRefractory(unsigned int n_reset, fptype* refactory_mass, inttype offset);
__global__ void MapResetThreaded(unsigned int n_reset, fptype* mass, fptype* refactory_mass, inttype ref_offset, unsigned int* rev_to, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts, unsigned int* map, fptype proportion);
__global__ void GetResetMass(unsigned int n_reset, fptype* sum, fptype* refactory_mass, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts);
__global__ void SumReset(unsigned int n_sum, fptype* sum, fptype* rate);
__global__ void Remap(int N, unsigned int* i_1, unsigned int t, unsigned int *map, unsigned int* first, unsigned int* length);
__global__ void ResetFinishThreaded(inttype n_reset, inttype* res_from, fptype* mass, inttype* map);
__global__ void CudaClearDerivative(inttype N, fptype* dydt, fptype* mass);
__global__ void CheckDerivativeEqualsZero(inttype N, fptype* derivative);
__device__ int modulo(int a, int b);

__global__ void CudaCalculateGridEfficacies(inttype N, fptype efficacy, fptype grid_cell_width, fptype* val, inttype* ia, inttype* ja);
#endif
