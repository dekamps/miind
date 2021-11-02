#include "CudaEuler.cuh"
#include <stdio.h>

__device__ int modulo(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

__global__ void CudaCalculateDerivative(inttype N, fptype rate, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int i_r = map[i + offset];
        fptype dr = 0.;
        for (unsigned int j = ia[i]; j < ia[i + 1]; j++) {
            int j_m = map[ja[j] + offset];
            dr += val[j] * mass[j_m];
        }
        dr -= mass[i_r];
        derivative[i_r] += rate * dr;
    }
}

__global__ void evolveMap(inttype N, inttype offset, inttype* map, inttype* unmap, inttype* cumulatives, 
    inttype* lengths, inttype _t) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_ind = i + offset;
        if (lengths[offset_ind] == 0)
            continue;
        int mapped = cumulatives[offset_ind] + modulo((i - cumulatives[offset_ind]) - _t, lengths[offset_ind]);
        map[offset_ind] = mapped + offset;
        unmap[mapped + offset] = offset_ind;
    }
}

__global__ void initCurand(curandState* state, unsigned long seed) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, index, 0, &state[index]);
}

__global__ void generatePoissonSpikes(inttype N, inttype offset, fptype rate, fptype timestep, inttype* spike_counts, curandState* state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        unsigned int r = curand_poisson(&state[index], rate* timestep);
        spike_counts[i+offset] = r;
    }
}

__global__ void CudaUpdateFiniteObjects(inttype N, inttype finite_offset, inttype* spike_counts, inttype* objects, fptype* refract_times,
    inttype* refract_inds, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype* unmap, inttype offset, curandState* state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        if (refract_times[i+finite_offset] > 0)
            continue;

        int current_index = unmap[objects[i+finite_offset]] - offset;
        for (unsigned int s = 0; s < spike_counts[i+finite_offset]; s++) {
            fptype r = curand_uniform(&state[index]);
            fptype check = 0.0;
            for (unsigned int j = ia[current_index]; j < ia[current_index + 1]; j++) {
                check += val[j];
                if (r < check) {
                    current_index = ja[j];
                    break;
                }
            }
        }
        
        objects[i+finite_offset] = map[current_index + offset];
    }
}

__global__ void CudaReversalFiniteObjects(inttype N, inttype offset, inttype* objects, inttype reversal_N, 
    unsigned int* rev_from, unsigned int* rev_to, inttype* map) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        // ew! We loop through all reversal cells for each object.
        for (unsigned int s = 0; s < reversal_N; s++) {
            if (map[rev_from[s]] != objects[i+offset])
                continue;

            objects[i+offset] = rev_to[s];
            break;
        }
    }
}

__global__ void countSpikesAndClear(inttype N, inttype finite_offset, inttype* spiked, inttype* total_spikes) {
    extern __shared__ inttype idata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) {
        idata[tid] = 0.;
        return;
    }


    idata[tid] = spiked[i + finite_offset];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            idata[tid] += idata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        total_spikes[blockIdx.x] = idata[0];
    }
}

__global__ void countSpikesAndClearSlow(inttype N, inttype finite_offset, inttype* spiked, inttype* total_spikes) {
    inttype total = 0;
    for (int i = 0; i < N; i++) {
        if (spiked[i + finite_offset] == 1)
            total++;
        spiked[i + finite_offset] = 0;
    }
    total_spikes[0] = total;
}

__global__ void CudaGridEvolveFiniteObjects(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times, fptype* val,
    inttype* ia, inttype* ja, inttype offset, curandState* state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        if (refract_times[i+finite_offset] > 0)
            continue;

        int current_index = objects[i+finite_offset] - offset;
        fptype r = curand_uniform(&state[index]);
        fptype check = 0.0;
        for (unsigned int j = ia[current_index]; j < ia[current_index + 1]; j++) {
            check += val[j];
            if (r < check) {
                objects[i+finite_offset] = ja[j] + offset;
                break;
            }
        }
    }
}

__global__ void CudaGridUpdateFiniteObjects(inttype N, inttype* spike_counts, inttype* objects,
    fptype* refract_times, inttype* refract_inds, fptype* stays, fptype* goes,
    int* offset1, int* offset2, inttype offset, curandState* state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        if (refract_times[i] > 0)
            continue;

        int current_index = objects[i] + offset;
        for (unsigned int s = 0; s < spike_counts[i]; s++) {
            fptype r = curand_uniform(&state[index]);
            if (r < stays[current_index]) {
                current_index = modulo(current_index + offset1[current_index], N);
            }
            else {
                current_index = modulo(current_index + offset2[current_index], N);
            }
        }

        objects[i] = current_index;
    }
}

__global__ void CudaGridUpdateFiniteObjectsCalc(inttype N, inttype finite_offset, inttype* spike_counts, inttype* objects,
    fptype* refract_times, inttype* refract_inds, fptype efficacy, fptype grid_cell_width, inttype grid_cell_offset, curandState* state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_index = i + finite_offset;
        if (refract_times[offset_index] > 0 || spike_counts[offset_index] == 0)
            continue;

        fptype spike_eff = spike_counts[offset_index] * efficacy;
        inttype ofs = (inttype)abs(spike_eff / grid_cell_width);
        fptype g = (fptype)fabs(spike_eff / grid_cell_width) - ofs;

        fptype r = curand_uniform(&state[index]);
        if (r > g) {
            int o1 = (spike_eff > 0 ? ofs : -ofs) * grid_cell_offset;
            objects[offset_index] = objects[offset_index] + o1;
        } else {
            int o2 = (spike_eff > 0 ? (ofs + 1) : -(ofs + 1))* grid_cell_offset;
            objects[offset_index] = objects[offset_index] + o2;
        }

    }
}

__global__ void CudaGridUpdateFiniteObjectsCalcNd(inttype N, inttype finite_offset, inttype* spike_counts, inttype* objects,
    fptype* refract_times, inttype* refract_inds, fptype* props, int* offsets, inttype proportion_stride, inttype grid_cell_offset, curandState* state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_index = i + finite_offset;
        if (refract_times[offset_index] > 0 || spike_counts[offset_index] == 0)
            continue;

        inttype obj = objects[offset_index];

        for (int s = 0; s < spike_counts[offset_index]; s++) {
            fptype r = curand_uniform(&state[index]);
            fptype total_prop = 0;
            for (int j = 0; j < proportion_stride; j++) {
                total_prop += props[(obj * proportion_stride) + j];
                if (r < total_prop) {
                    obj = obj + offsets[(obj * proportion_stride) + j];
                    break;
                }
            }
        }   

        objects[offset_index] = obj;
    }
}

__global__ void CudaSolveIzhikevichNeurons(inttype N, inttype* spike_counts, inttype* spiked, fptype* vs, fptype* ws,
    fptype* refract_times, fptype refractory_time, fptype timestep, curandState* state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        spiked[i] = 0;

        if (refract_times[i] >= 0) {
            refract_times[i] -= timestep;
            if (refract_times[i] < 0) {
                refract_times[i] = -1.0;
                vs[i] = -50.0;
                ws[i] += 2.0;
            }
        }
        else {

            vs[i] += spike_counts[i] * 0.5;

            fptype v = vs[i];
            fptype w = ws[i];

            if (v > -30.0) {
                refract_times[i] = refractory_time;
                spiked[i] = 1;
            } else {
                fptype v_prime = (0.04 * (v * v)) + (5 * v) + 140 - w;
                fptype w_prime = 0.02 * ((0.2 * v) - w);

                vs[i] = vs[i] + (timestep * 1000 * v_prime);
                ws[i] = ws[i] + (timestep * 1000 * w_prime);
            }
        }
    }
}

__global__ void CudaGridResetFiniteObjects(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times,
    inttype* refract_inds, inttype threshold_col_index, inttype reset_col_index, inttype reset_w_rows, 
    inttype res_v, fptype res_v_stays, fptype refractory_time, fptype timestep, inttype* spiked, inttype offset, curandState* state, inttype num_cells) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_ind = i + finite_offset;
        spiked[offset_ind] = 0;

        if (refract_times[offset_ind] >= 0) {
            refract_times[offset_ind] -= timestep;

            if (refract_times[offset_ind] <= 0) {
                refract_times[offset_ind] = -1.0;

                objects[offset_ind] = refract_inds[offset_ind] + offset;
            }
        }
        else {
            // Is the column of this cell above the threshold column?
            inttype i_col = ((objects[offset_ind]-offset) % res_v);
            if (i_col >= threshold_col_index) {

                //roll for whether we're in the quoted reset cell or the cell above
                fptype r = curand_uniform(&state[index]);
                if (r > res_v_stays)
                    if (reset_w_rows > 0)
                        reset_w_rows = reset_w_rows + 1;
                    else
                        reset_w_rows = reset_w_rows - 1;

                // Calculate the reset cell
                inttype reset_cell = modulo(((objects[offset_ind]-offset) - (i_col - reset_col_index)) + (reset_w_rows * res_v), num_cells);

                refract_times[offset_ind] = refractory_time;
                refract_inds[offset_ind] = reset_cell; // instead of storing the current threshold cell - store the target reset cell
                spiked[offset_ind] = 1;
            }
        }
    }
}

// For meshes which have the strips going up instead of across
__global__ void CudaGridResetFiniteObjectsRot(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times,
    inttype* refract_inds, inttype threshold_col_index, inttype reset_col_index, inttype reset_w_rows,
    inttype res_w, fptype res_v_stays, fptype refractory_time, fptype timestep, inttype* spiked, inttype offset, curandState* state, inttype num_cells) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_ind = i + finite_offset;
        spiked[offset_ind] = 0;

        if (refract_times[offset_ind] >= 0) {
            refract_times[offset_ind] -= timestep;

            if (refract_times[offset_ind] <= 0) {
                refract_times[offset_ind] = -1.0;

                objects[offset_ind] = refract_inds[offset_ind] + offset;
            }
        }
        else {
            // Is the column of this cell above the threshold column?
            inttype i_col = int((objects[offset_ind] - offset) / res_w);

            if (i_col >= threshold_col_index) {

                //roll for whether we're in the quoted reset cell or the cell above
                fptype r = curand_uniform(&state[index]);
                if (r > res_v_stays)
                    if (reset_w_rows > 0)
                        reset_w_rows = reset_w_rows + 1;
                    else
                        reset_w_rows = reset_w_rows - 1;

                // Calculate the reset cell
                inttype reset_cell = modulo(((objects[offset_ind] - offset) + reset_w_rows ) - ((i_col - reset_col_index) * res_w), num_cells);

                refract_times[offset_ind] = refractory_time;
                refract_inds[offset_ind] = reset_cell; // instead of storing the current threshold cell - store the target reset cell
                spiked[offset_ind] = 1;
            }
        }
    }
}

__global__ void CudaResetFiniteObjects(inttype N, inttype offset, inttype* objects, fptype* refract_times, 
    inttype* refract_inds, fptype refractory_time, inttype reset_N,
    unsigned int* rev_from, inttype* unmap, inttype* spiked) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_ind = i + offset;
        spiked[offset_ind] = 0;
        // ew! We loop through all reset cells for each object.
        for (unsigned int s = 0; s < reset_N; s++) {
            if (rev_from[s] != unmap[objects[offset_ind]])
                continue;

            refract_times[offset_ind] = refractory_time;
            refract_inds[offset_ind] = rev_from[s];
            spiked[offset_ind] = 1;
            break;
        }
    }
}

__global__ void CudaCheckRefractingFiniteObjects(inttype N, inttype finite_offset, inttype* objects, fptype* refract_times,
    inttype* refract_inds, fptype timestep, inttype reset_N,
    unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, curandState* state, inttype* map, inttype* unmap) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int offset_ind = i + finite_offset;
        if (refract_times[offset_ind] >= 0) {
            refract_times[offset_ind] -= timestep;

            if (refract_times[offset_ind] <= 0) {
                refract_times[offset_ind] = -1.0;

                fptype r = curand_uniform(&state[index]);
                fptype check = 0.0;
                // ew! We loop through all reset cells for each object.
                for (unsigned int s = 0; s < reset_N; s++) {
                    if (rev_from[s] != refract_inds[offset_ind])
                        continue;

                    check += rev_alpha[s];
                    if (r < check) {
                        objects[offset_ind] = map[rev_to[s]];
                        break;
                    }
                }
            }

        }
        
    }
}

__global__ void CudaSingleTransformStep(inttype N, fptype* derivative, fptype* mass, fptype* val, inttype* ia, inttype* ja, inttype* map, inttype offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int i_r = map[i + offset];
        fptype dr = 0.;
        for (unsigned int j = ia[i]; j < ia[i + 1]; j++) {
            int j_m = map[ja[j] + offset];
            dr += val[j] * mass[j_m];
        }
        dr -= mass[i_r];
        derivative[i_r] += dr;
    }
}


// Performing this calculation per iteration doubles the simulation time.
// This function shouldn't be used in that way but it a good example in case you want to
// include some kind of time dependent efficacy.
__global__ void CudaCalculateGridEfficacies(inttype N,
    fptype efficacy, fptype grid_cell_width, inttype grid_offset_width,
    fptype* stays, fptype* goes, int* offset1s, int* offset2s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        inttype ofs = (inttype)abs(efficacy / grid_cell_width);
        fptype g = (fptype)fabs(efficacy / grid_cell_width) - ofs;
        fptype s = 1.0 - g;

        int o1 = (efficacy > 0 ? ofs : -ofs) * grid_offset_width;
        int o2 = (efficacy > 0 ? (ofs + 1) : -(ofs + 1)) * grid_offset_width;

        stays[modulo(i + o1, N)] = s;
        goes[modulo(i + o2, N)] = g;
        offset1s[modulo(i + o1, N)] = -o1;
        offset2s[modulo(i + o2, N)] = -o2;
    }
}

// Performing this calculation per iteration doubles the simulation time.
// This function shouldn't be used in that way but it a good example in case you want to
// include some kind of time dependent efficacy.
__global__ void CudaCalculateGridCellEfficacies(inttype N,
    fptype* cell_vals, fptype grid_cell_width, inttype grid_offset_width,
    fptype* stays, fptype* goes, int* offset1s, int* offset2s, inttype vs_offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        inttype ofs = (inttype)abs(cell_vals[i + vs_offset] / grid_cell_width);
        fptype g = (fptype)fabs(cell_vals[i + vs_offset] / grid_cell_width) - ofs;
        fptype s = 1.0 - g;

        int o1 = (cell_vals[i + vs_offset] > 0 ? ofs : -ofs) * grid_offset_width;
        int o2 = (cell_vals[i + vs_offset] > 0 ? (ofs + 1) : -(ofs + 1)) * grid_offset_width;

        stays[modulo(i + o1, N)] = s;
        goes[modulo(i + o2, N)] = g;
        offset1s[modulo(i + o1, N)] = -o1;
        offset2s[modulo(i + o2, N)] = -o2;
    }
}

// As above, this function should not be used per iteration as the efficacy doesn't
// change during simulation.
__global__ void CudaCalculateGridEfficaciesWithConductance(inttype N,
    fptype efficacy, fptype grid_cell_width, inttype grid_cell_offset, fptype* cell_vs, fptype cond_stable,
    fptype* stays, fptype* goes, int* offset1s, int* offset2s, inttype vs_offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        // WARNING! If the offset changes from, say, 0 and 1 to 1 and 2 along
        // a strip (due to changing V), there will be an overwrite of goes and stays
        // which will lead to mass loss.
        // Solution must be to allow multiple stays and goes into each cell. There
        // should be a github bug report for this.
        fptype eff = efficacy * (cell_vs[i + vs_offset] - cond_stable);
        inttype ofs = (inttype)abs(eff / grid_cell_width);
        fptype g = (fptype)fabs(eff / grid_cell_width) - ofs;
        fptype s = 1.0 - g;

        int o1 = (efficacy > 0 ? ofs : -ofs) * grid_cell_offset;
        int o2 = (efficacy > 0 ? (ofs + 1) : -(ofs + 1)) * grid_cell_offset;

        stays[modulo(i + o1, N)] = s;
        goes[modulo(i + o2, N)] = g;
        offset1s[modulo(i + o1, N)] = -o1;
        offset2s[modulo(i + o2, N)] = -o2;
    }
}

__global__ void CudaCalculateGridDerivative(inttype N, fptype rate, fptype stays,
    fptype goes, int offset_1, int offset_2,
    fptype* derivative, fptype* mass, inttype offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int io = i + offset;
        fptype dr = 0.;

        dr += stays * mass[(modulo(io + offset_1, N)) + offset];
        dr += goes * mass[(modulo(io + offset_2, N)) + offset];
        dr -= mass[io];
        derivative[io] += rate * dr;
    }
}

__global__ void CudaCalculateGridDerivativeWithEfficacy(inttype N, fptype rate, fptype* stays,
    fptype* goes, int* offset_1, int* offset_2,
    fptype* derivative, fptype* mass, inttype offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int io = i + offset;
        fptype dr = 0.;
        dr += stays[i] * mass[(modulo(io + offset_1[i], N)) + offset];
        dr += goes[i] * mass[(modulo(io + offset_2[i], N)) + offset];
        dr -= mass[io];
        derivative[io] += rate * dr;
    }
}

// Instead of stays and goes and two offsets, 
// Have a proportion array and an offset array each of size proportion_stride
// Allows for more customisable transitions.
// For example, with proportion_stride = 2^num_dimensions we can define the transition due
// to an efficacy jump in any direction. See 3D tsodyks example.
__global__ void CudaCalculateGridDerivativeWithEfficacyNd(inttype N, fptype rate, 
    fptype* props, int* offsets, inttype proportion_stride,
    fptype* derivative, fptype* mass, inttype offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        int io = i + offset;
        fptype dr = 0.;
        for (int j = 0; j < proportion_stride; j++) {
            dr += props[(i* proportion_stride)+j] * mass[(modulo(io + offsets[(i * proportion_stride) +j], N)) + offset];
        }
        dr -= mass[io];
        derivative[io] += rate * dr;
    }
}

__global__ void EulerStep(inttype N, fptype* derivative, fptype* mass, fptype timestep)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride) {
        mass[i] += derivative[i] * timestep;
    }
}

__global__ void CheckDerivativeEqualsZero(inttype N, fptype* derivative)
{
    fptype total = 0.;

    for (int i = 0; i < N; i++) {
        total += derivative[i];
        printf("add : %i %f : %f\n", i, derivative[i], total);

    }

    printf("data : %f\n", total);
}

__global__ void MapReversal(unsigned int n_reversal, unsigned int* rev_from, unsigned int* rev_to, fptype* rev_alpha, fptype* mass, unsigned int* map)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n_reversal; i += stride) {
        fptype m = mass[map[rev_from[i]]];
        mass[map[rev_from[i]]] = 0.;
        mass[map[rev_to[i]]] += m;
    }
}

__global__ void MapResetToRefractory(unsigned int n_reset, unsigned int* res_from, fptype* mass, unsigned int* map, fptype* refactory_mass) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n_reset; i += stride) {
        refactory_mass[i] = mass[map[res_from[i]]];
    }
}

__global__ void MapResetShiftRefractory(unsigned int n_reset, fptype* refactory_mass, inttype offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = offset + index; i < offset + n_reset; i += stride) {
        refactory_mass[i + n_reset] = refactory_mass[i];
    }
}

__global__ void MapResetThreaded(unsigned int n_reset, fptype* mass, fptype* refactory_mass, inttype ref_offset,
    unsigned int* rev_to, fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts, unsigned int* map, fptype proportion) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n_reset; i += stride) {
        int i_r = map[rev_to[i]];
        fptype dr = 0.;
        for (unsigned int j = rev_offsets[i]; j < rev_offsets[i] + rev_counts[i]; j++) {
            dr += rev_alpha[j] * refactory_mass[ref_offset + j] * proportion;

        }
        mass[i_r] += dr;
    }
}

__global__ void GetResetMass(unsigned int n_reset, fptype* sum, fptype* refactory_mass,
    fptype* rev_alpha, inttype* rev_offsets, inttype* rev_counts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n_reset; i += stride) {
        fptype dr = 0.;
        for (unsigned int j = rev_offsets[i]; j < rev_offsets[i] + rev_counts[i]; j++) {
            dr += rev_alpha[j] * refactory_mass[j];
        }
        sum[i] = dr;
    }
}

__global__ void SumReset(unsigned int n_sum, fptype* sum, fptype* rate) {
    extern __shared__ fptype sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_sum) {
        sdata[tid] = 0.;
        return;
    }


    sdata[tid] = sum[i];
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
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

__global__ void ResetFinishThreaded(inttype n_reset, inttype* res_from, fptype* mass, inttype* map) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n_reset; i += stride) {
        mass[map[res_from[i]]] = 0.;
    }
}

__global__ void Remap(int N, unsigned int* i_1, unsigned int t, unsigned int* map, unsigned int* first, unsigned int* length)
{
    unsigned int i_start = *i_1;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
        if (i >= i_start)
            map[i] = modulo((i - t - first[i]), length[i]) + first[i];
}

__global__ void CudaClearDerivative(inttype N, fptype* dydt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        dydt[i] = 0.;
}

__global__ void CudaClearSpikeCounts(inttype N, inttype* dydt) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        dydt[i] = 0;
}
