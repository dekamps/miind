#include "CudaTest.cuh"

__device__ int modulo(int a, int b){
        int r = a%b;
        return r< 0 ? r + b : r;
}

__global__ void CalculateDerivative(unsigned int N, fptype* derivative, fptype* mass, fptype* val, unsigned int* ia, unsigned int* ja, unsigned int* map){
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride ){
      derivative[map[i]]  = -mass[map[i]];
      for(unsigned int j = ia[i]; j < ia[i+1]; j++)
          derivative[map[i]] += val[j]*mass[map[ja[j]]];
    }
}

__global__ void EulerStep(unsigned int N, unsigned int n_div, fptype* derivative, fptype* mass, fptype timestep, fptype rate)
{
    fptype multiplier = rate*timestep/static_cast<fptype>(n_div);
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride ){
       mass[i] += derivative[i]*multiplier;
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

__global__ void MapReset(unsigned int n_reset, unsigned int* res_from, unsigned int* res_to, fptype* res_alpha, fptype* mass, unsigned int* map, float* rate)
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
  
__global__ void MapFinish(unsigned int n_reset, unsigned int* res_from, fptype* mass, unsigned int* map){
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

