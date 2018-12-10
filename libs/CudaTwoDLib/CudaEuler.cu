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

__global__ void CudaSingleTransformStep(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset)
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
      derivative[i_r] += dr;
    }
}

__global__ void CudaCalculateGridDerivative(inttype N, fptype rate, fptype stays,
  fptype goes, int offset_1, int offset_2,
  fptype* derivative, fptype* mass, inttype offset)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride ){
      int io = i+offset;
      fptype dr = 0.;
      dr += stays*mass[(modulo(io+offset_1,N))+offset];
  		dr += goes*mass[(modulo(io+offset_2,N))+offset];
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

__global__ void MapResetToRefractory(unsigned int n_reset, unsigned int* res_from, fptype* mass, unsigned int* map, fptype* refactory_mass){
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n_reset; i += stride){
    refactory_mass[i] = mass[map[res_from[i]]];
  }
}

__global__ void MapResetShiftRefractory(unsigned int n_reset, fptype* refactory_mass, inttype offset){
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // printf("%i %i %i\n", offset+index, offset+n_reset, n_reset);
  for (int i = offset+index; i < offset+n_reset; i+=stride){
    refactory_mass[i+n_reset] = refactory_mass[i];
  }
}

__global__ void MapResetThreaded(unsigned int n_reset, fptype* sum, fptype* mass, fptype* refactory_mass, inttype ref_offset,
 unsigned int* rev_to, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts, unsigned int* map){
   int index  = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int i = index; i < n_reset; i+= stride ){
     int i_r = map[rev_to[i]];
     fptype dr = 0.;
     for(unsigned int j = rev_offsets[i]; j < rev_offsets[i]+rev_counts[i]; j++){
         dr += rev_alpha[j]*refactory_mass[ref_offset+j];

     }
     sum[i] += dr;
     mass[i_r] += dr;
   }
}

 __global__ void SumReset(unsigned int n_sum, fptype* sum, fptype* rate){
   extern __shared__ fptype sdata[];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i >= n_sum)
    return;

  sdata[tid] = sum[i];
  __syncthreads();
  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) {
    rate[blockIdx.x] += sdata[0];
  }
 }

__global__ void ResetFinishThreaded(inttype n_reset, inttype* res_from, fptype* mass, inttype* map){
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n_reset; i+= stride)
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
