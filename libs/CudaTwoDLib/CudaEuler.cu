#include "CudaEuler.cuh"
#include <stdio.h>
__device__ int modulo(int a, int b){
        int r = a%b;
        return r< 0 ? r + b : r;
}

__global__ void CudaCalculateDerivative(inttype N, fptype rate, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride ){
      int i_r = map[i+offset];
      fptype dr = 0.;
      for(unsigned int j = ia[i]; j < ia[i+1]; j++){
          int j_m = map[ja[j]+offset];
          dr += val[j]*mass[j_m];
      }
      dr -= mass[i_r];
      derivative[i_r] += rate*dr;
    }
}

__global__ void CudaSingleTransformStep(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset, inttype workingN, inttype* workindex)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < workingN; i+= stride ){
      int i_r = map[workindex[i+offset]];
      fptype dr = 0.;
      for(unsigned int j = ia[workindex[i+offset]-offset]; j < ia[workindex[i+offset]-offset+1]; j++){
          int j_m = map[ja[j]];
          dr += val[j]*mass[j_m];
      }
      dr -= mass[i_r];
      derivative[i_r] += dr;
    }
}

__global__ void CudaCalculateGridDerivative(inttype N, fptype rate, fptype stays,
  fptype goes, inttype offset_1, inttype offset_2,
  fptype* derivative, fptype* mass, inttype offset, inttype workingN, inttype* working)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < workingN; i+= stride ){
      fptype dr = 0.;
      dr += stays*mass[((((working[i+offset]+offset_1)%N)+N) % N)];
  		dr += goes*mass[((((working[i+offset]+offset_2)%N)+N) % N)];

      int io = working[i+offset];
      dr -= mass[io];
      derivative[io] += rate*dr;
    }
}

__global__ void EulerStep(inttype N, fptype* derivative, fptype* mass, fptype timestep)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride ){
       mass[i] += derivative[i]*timestep;
    }
}

__global__ void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, fptype* mass, unsigned int* map)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n_reversal; i+= stride){
        fptype m = mass[map[rev_from[i]]];
        mass[map[rev_from[i]]] = 0.;
        mass[map[rev_to[i]]] += m;
    }
}

__global__ void MapReset(inttype n_reset, inttype* res_from, inttype* res_to, fptype* res_alpha, fptype* mass, inttype* map, fptype* rate)
{
  // run as a single kernel, this function does reduction of the firing rate
  fptype sum = 0;
  for (int i = 0; i < n_reset; i++){
    fptype m = res_alpha[i]*mass[map[res_from[i]]];
    mass[map[res_to[i]]] += m;
    sum += m;
  }
  *rate = sum;
 }

__global__ void ResetFinish(inttype n_reset, inttype* res_from, fptype* mass,  inttype* map){
   for (int i = 0; i < n_reset; i++)
      mass[map[res_from[i]]] = 0.;
}

__global__ void Remap(int N, unsigned int* i_1, unsigned int t, unsigned int *map, unsigned int* first, unsigned int* length)
{
  unsigned int i_start = *i_1;

  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride)
      if (i >= i_start)
          map[i] =  modulo((i - t - first[i]),length[i]) + first[i];
}

__global__ void CudaClearDerivative(inttype N, fptype* dydt, fptype* mass){
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i+= stride)
        dydt[i]  = 0.;
}
