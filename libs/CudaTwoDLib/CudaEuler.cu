#include "CudaEuler.cuh"
#include <stdio.h>

__global__ void CudaSingleTransformStepIndexed(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset, inttype workingN, inttype* workindex)
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

__global__ void CudaSingleTransformStepBound(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia,
  inttype* ja, inttype* map, inttype offset, inttype sx, inttype ex)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = sx+index; i < ex; i+= stride ){
      int i_r = map[i];
      fptype dr = 0.;
      for(unsigned int j = ia[i-offset]; j < ia[i-offset+1]; j++){
          int j_m = map[ja[j]];
          dr += val[j]*mass[j_m];
      }
      dr -= mass[i_r];
      derivative[i_r] += dr;
    }
}

__global__ void CudaCalculateGridDerivativeIndexed(inttype N, fptype rate, fptype stays,
  fptype goes, inttype offset_1, inttype offset_2,
  fptype* derivative, fptype* mass, inttype offset, inttype workingN, inttype* working)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < workingN; i+= stride ){
      int io = working[i+offset];
      fptype dr = 0.;
      dr += stays*mass[((((io+offset_1)%N)+N) % N)];
  		dr += goes*mass[((((io+offset_2)%N)+N) % N)];
      dr -= mass[io];
      derivative[io] += rate*dr;
    }
}

__global__ void CudaCalculateGridDerivativeBound(inttype N, fptype rate, fptype stays,
  fptype goes, inttype offset_1, inttype offset_2,
  fptype* derivative, fptype* mass, inttype offset, inttype sx, inttype ex)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = sx+index; i < ex; i+= stride ){
      fptype dr = 0.;
      dr += stays*mass[((((i+offset_1)%N)+N) % N)];
  		dr += goes*mass[((((i+offset_2)%N)+N) % N)];

      dr -= mass[i];
      derivative[i] += rate*dr;
    }
}

__global__ void EulerStepIndexed(fptype* derivative, fptype* mass, inttype offset, inttype workingN, inttype* workindex)
{
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < workingN; i+= stride ){
    mass[workindex[i+offset]] += derivative[workindex[i+offset]];
  }
}

__global__ void EulerStepBound(fptype* derivative, fptype* mass, inttype sx, inttype ex)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = sx+index; i < ex; i+= stride ){
       mass[i] += derivative[i];
    }
}

 __global__ void MapResetIndexed(inttype n_reset, inttype* res_from, inttype* res_to, fptype* res_alpha, fptype* mass, inttype* map, fptype* rate, inttype workingN, inttype* workindex)
 {
   // run as a single kernel, this function does reduction of the firing rate
   fptype sum = 0;
   for (int i = 0; i < workingN; i++){
     fptype m = res_alpha[workindex[i]]*mass[map[res_from[workindex[i]]]];
     mass[map[res_to[workindex[i]]]] += m;
     sum += m;
   }
   *rate = sum;
  }

__global__ void ResetFinishIndexed(inttype n_reset, inttype* res_from, fptype* mass,  inttype* map, inttype workingN, inttype* workindex){
   for (int i = 0; i < workingN; i++)
      mass[map[res_from[workindex[i]]]] = 0.;
}

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
  fptype goes, inttype offset_1, inttype offset_2,
  fptype* derivative, fptype* mass, inttype offset)
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride ){
      int io = i+offset;
      fptype dr = 0.;
      dr += stays*mass[((((io+offset_1)%N)+N) % N)+offset];
  		dr += goes*mass[((((io+offset_2)%N)+N) % N)+offset];
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

 __global__ void MapResetThreaded(unsigned int N, fptype* sum, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, unsigned int* map, inttype offset)
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
     sum[i] += dr;
     derivative[i_r] += dr;
   }
 }

 __global__ void SumReset(unsigned int n_sum, fptype* sum, fptype* rate){
   extern __shared__ fptype sdata[];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
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

__global__ void ResetFinish(inttype n_reset, inttype* res_from, fptype* mass,  inttype* map){
   for (int i = 0; i < n_reset; i++)
      mass[map[res_from[i]]] = 0.;
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
