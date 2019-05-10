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


// Performing this calculation per iteration doubles the simulation time.
// This function shouldn't be used in that way but it a good example in case you want to
// include some kind of variable efficacy.
__global__ void CudaCalculateGridEfficacies(inttype N,
  fptype efficacy, fptype grid_cell_width,
  fptype* stays, fptype* goes, int* offset1s, int* offset2s)
{
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i+= stride ){
    inttype ofs = (inttype)abs(efficacy / grid_cell_width);
    fptype g = (fptype)fabs(efficacy / grid_cell_width) - ofs;
    fptype s = 1.0 - g;

    int o1 = efficacy > 0 ? ofs : -ofs;
    int o2 = efficacy > 0 ? (ofs+1) : (ofs-1);

    stays[modulo(i+o1,N)] = s;
    goes[modulo(i+o2,N)] = g;
    offset1s[modulo(i+o1,N)] = -o1;
    offset2s[modulo(i+o2,N)] = -o2;
  }
}

// As above, this function should not be used per iteration as the efficacy doesn't
// change during simulation.
__global__ void CudaCalculateGridEfficaciesWithConductance(inttype N,
  fptype efficacy, fptype grid_cell_width, fptype* cell_vs, fptype cond_stable,
  fptype* stays, fptype* goes, int* offset1s, int* offset2s, inttype vs_offset)
{
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i+= stride ){
    // WARNING! If the offset changes from, say, 0 and 1 to 1 and 2 along
    // a strip (due to changing V), there will be an overwrite of goes and stays
    // which will lead to mass loss.
    // Solution must be to allow multiple stays and goes into each cell. There
    // should be a github bug report for this.
    fptype eff = efficacy * (cell_vs[i + vs_offset] - cond_stable);
    inttype ofs = (inttype)abs(eff / grid_cell_width);
    fptype g = (fptype)fabs(eff / grid_cell_width) - ofs;
    fptype s = 1.0 - g;

    int o1 = efficacy > 0 ? ofs : -ofs;
    int o2 = efficacy > 0 ? (ofs+1) : (ofs-1);

    stays[modulo(i+o1,N)] = s;
    goes[modulo(i+o2,N)] = g;
    offset1s[modulo(i+o1,N)] = -o1;
    offset2s[modulo(i+o2,N)] = -o2;
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

__global__ void CudaCalculateGridDerivativeWithEfficacy(inttype N, fptype rate, fptype* stays,
  fptype* goes, int* offset_1, int* offset_2,
  fptype* derivative, fptype* mass, inttype offset)
{
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i+= stride ){
    int io = i+offset;
    fptype dr = 0.;
    dr += stays[i]*mass[(modulo(io+offset_1[i],N))+offset];
    dr += goes[i]*mass[(modulo(io+offset_2[i],N))+offset];
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

__global__ void CheckDerivativeEqualsZero(inttype N, fptype* derivative)
{
  fptype total = 0.;

  for (int i = 0; i < N; i++){
     total += derivative[i];
     printf("add : %i %f : %f\n", i, derivative[i], total);

  }

  printf("data : %f\n", total);
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

  for (int i = offset+index; i < offset+n_reset; i+=stride){
    refactory_mass[i+n_reset] = refactory_mass[i];
  }
}

__global__ void MapResetThreaded(unsigned int n_reset, fptype* mass, fptype* refactory_mass, inttype ref_offset,
 unsigned int* rev_to, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts, unsigned int* map, fptype proportion){
   int index  = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int i = index; i < n_reset; i+= stride ){
     int i_r = map[rev_to[i]];
     fptype dr = 0.;
     for(unsigned int j = rev_offsets[i]; j < rev_offsets[i]+rev_counts[i]; j++){
         dr += rev_alpha[j]*refactory_mass[ref_offset+j]*proportion;

     }
     mass[i_r] += dr;
   }
}

__global__ void GetResetMass(unsigned int n_reset, fptype* sum, fptype* refactory_mass,
 fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts){
   int index  = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   for (int i = index; i < n_reset; i+= stride ){
     fptype dr = 0.;
     for(unsigned int j = rev_offsets[i]; j < rev_offsets[i]+rev_counts[i]; j++){
         dr += rev_alpha[j]*refactory_mass[j];
     }
     sum[i] = dr;
   }
}

 __global__ void SumReset(unsigned int n_sum, fptype* sum, fptype* rate){
   extern __shared__ fptype sdata[];
  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  if(i >= n_sum){
    sdata[tid] = 0.;
    return;
  }


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
    rate[blockIdx.x] = sdata[0];
  }
 }

__global__ void ResetFinishThreaded(inttype n_reset, inttype* res_from, fptype* mass, inttype* map){
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n_reset; i+= stride){
    mass[map[res_from[i]]] = 0.;
  }
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
