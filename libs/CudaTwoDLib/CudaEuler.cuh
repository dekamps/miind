#ifndef CUEMTESTCUH
#define CUEMTESTCUH

#include <curand.h>
#include <curand_kernel.h>

typedef unsigned int inttype;
typedef float fptype;

__global__ void CudaCalculateDerivative(inttype N, fptype rate, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset);
__global__ void CudaSingleTransformStep(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset);
__global__ void CudaCalculateGridDerivative(inttype N, fptype rate, fptype stays, fptype goes, int offset_1, int offset_2, fptype* derivative, fptype* mass, inttype offset);
__global__ void EulerStep(inttype N, fptype* derivative, fptype* mass, fptype timestep);
__global__ void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, fptype* mass, unsigned int* map);
__global__ void MapResetToRefractory(unsigned int n_reset, unsigned int* res_from, fptype* mass, unsigned int* map, fptype* refactory_mass);
__global__ void MapResetShiftRefractory(unsigned int n_reset, fptype* refactory_mass, inttype offset);
__global__ void MapResetThreaded(unsigned int n_reset, fptype* mass, fptype* refactory_mass, inttype ref_offset, unsigned int* rev_to, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts, unsigned int* map, fptype proportion);
__global__ void GetResetMass(unsigned int n_reset, fptype* sum, fptype* refactory_mass, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts);
__global__ void SumReset(unsigned int n_sum, fptype* sum, fptype* rate);
__global__ void Remap(int N, unsigned int* i_1, unsigned int t, unsigned int* map, unsigned int* first, unsigned int* length);
__global__ void ResetFinishThreaded(inttype n_reset, inttype* res_from, fptype* mass, inttype* map);
__global__ void CudaClearDerivative(inttype N, fptype* dydt);
__global__ void CheckDerivativeEqualsZero(inttype N, fptype* derivative);
__device__ int modulo(int a, int b);

// Finite Size Functions
__global__ void countSpikesAndClear(inttype N, inttype finite_offset, inttype* spiked, inttype* total_spikes);
__global__ void evolveMap(inttype N, inttype offset, inttype* map, inttype* unmap, inttype* cumulatives, inttype* lengths, inttype _t);
__global__ void initCurand(curandState* state, unsigned long seed);
__global__ void generatePoissonSpikes(inttype N, inttype offset, fptype rate, fptype timestep, inttype* spike_counts, curandState* state);
__global__ void CudaUpdateFiniteObjects(inttype N, inttype finite_offset, inttype* spike_counts, inttype* objects, fptype* refract_times, inttype* refract_inds, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype* unmap, inttype offset, curandState* state);
__global__ void CudaReversalFiniteObjects(inttype N, inttype offset, inttype* objects, inttype reversal_N, unsigned int* rev_from, unsigned int* rev_to, inttype* map);
__global__ void CudaResetFiniteObjects(inttype N, inttype offset, inttype* objects, fptype* refract_times, inttype* refract_inds, fptype refractory_time, inttype reset_N, unsigned int* rev_from, inttype* unmap, inttype* spiked);
__global__ void CudaCheckRefractingFiniteObjects(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times, inttype* refract_inds, fptype timestep, inttype reset_N, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, curandState* state, inttype* map, inttype* unmap);

__global__ void CudaGridEvolveFiniteObjects(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times, fptype* val, inttype* ia, inttype* ja, inttype offset, curandState* state);
__global__ void CudaGridUpdateFiniteObjects(inttype N, inttype* spike_counts, inttype* objects, fptype* refract_times, inttype* refract_inds, fptype* stays, fptype* goes, int* offset1, int* offset2, inttype offset, curandState* state);
__global__ void CudaGridUpdateFiniteObjectsCalc(inttype N, inttype finite_offset, inttype* spike_counts, inttype* objects,
    fptype* refract_times, inttype* refract_inds, fptype efficacy, fptype grid_cell_width, inttype grid_cell_offset, curandState* state);
__global__ void CudaGridUpdateFiniteObjectsCalcNd(inttype N, inttype finite_offset, inttype* spike_counts, inttype* objects,
    fptype* refract_times, inttype* refract_inds, fptype* props, int* offsets, inttype proportion_stride, inttype grid_cell_offset, curandState* state);
__global__ void CudaGridResetFiniteObjects(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times,
    inttype* refract_inds, inttype threshold_col_index, inttype reset_col_index, inttype reset_w_rows,
    inttype res_v, fptype res_v_stays, fptype refractory_time, fptype timestep, inttype* spiked, inttype offset, curandState* state, inttype num_cells);
__global__ void CudaGridResetFiniteObjectsRot(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times,
    inttype* refract_inds, inttype threshold_col_index, inttype reset_col_index, inttype reset_w_rows,
    inttype res_v, fptype res_v_stays, fptype refractory_time, fptype timestep, inttype* spiked, inttype offset, curandState* state, inttype num_cells);

__global__ void countSpikesAndClearSlow(inttype N, inttype finite_offset, inttype* spiked, inttype* total_spikes);
__global__ void CudaClearSpikeCounts(inttype N, inttype* dydt);

__global__ void CudaSolveIzhikevichNeurons(inttype N, inttype* spike_counts, inttype* spiked, fptype* vs, fptype* ws, fptype* refract_times, fptype refractory_time, fptype timestep, curandState* state);

// Grid Algorithm Specialisations
__global__ void CudaCalculateGridDerivativeWithEfficacy(inttype N, fptype rate, fptype* stays, fptype* goes, int* offset_1, int* offset_2, fptype* derivative, fptype* mass, inttype offset);
__global__ void CudaCalculateGridDerivativeWithEfficacyNd(inttype N, fptype rate, fptype* props, int* offsets, inttype proportion_stride, fptype* derivative, fptype* mass, inttype offset);
__global__ void CudaCalculateGridEfficaciesWithConductance(inttype N, fptype efficacy, fptype grid_cell_width, inttype grid_cell_offset, fptype* cell_vs, fptype cond_stable, fptype* stays, fptype* goes, int* offset1s, int* offset2s, inttype vs_offset);
__global__ void CudaCalculateGridEfficacies(inttype N, fptype efficacy, fptype grid_cell_width, inttype grid_offset_width, fptype* stays, fptype* goes, int* offset1s, int* offset2s);
__global__ void CudaCalculateGridCellEfficacies(inttype N, fptype* cell_vals, fptype grid_cell_width, inttype grid_offset_width, fptype* stays, fptype* goes, int* offset1s, int* offset2s, inttype vs_offset);
#endif
