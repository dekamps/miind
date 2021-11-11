// Copyright (c) 2005 - 2015 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "CudaEuler.cuh"
#include "CudaOde2DSystemAdapter.cuh"

using namespace CudaTwoDLib;

namespace {
	const float tolerance = 1e-6;
}

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

CudaOde2DSystemAdapter::CudaOde2DSystemAdapter
(
    TwoDLib::Ode2DSystemGroup& group,
		MPILib::Time network_time_step,
	unsigned int mesh_objects_start_index
):
_group(group),
_mesh_objects_start_index(mesh_objects_start_index),
_time_step(group.MeshObjects()[0].TimeStep()),
_network_time_step(network_time_step),
_mesh_size(group.MeshObjects().size()),
_n(group.Mass().size()),
_hostmass(_n,0.),
_hostmap(_n,0.),
_hostunmap(_n, 0.),
_host_map_cumulative_value(_n, 0),
_host_map_strip_length_value(_n, 0),
_offsets(group.Offsets()),
_nr_refractory_steps(group.MeshObjects().size(),0),
_refractory_prop(group.MeshObjects().size(),0),
_refractory_mass(group.MeshObjects().size(),0),
_refractory_mass_local(group.MeshObjects().size()),
_nr_minimal_resets(_group.MeshObjects().size(),0),
_res_to_minimal(_group.MeshObjects().size(),0),
_res_from_ordered(_group.MeshObjects().size(),0),
_res_to_ordered(_group.MeshObjects().size(), 0),
_res_alpha_ordered(_group.MeshObjects().size(),0),
_res_from_counts(_group.MeshObjects().size(),0),
_res_from_offsets(_group.MeshObjects().size(),0),
_vec_alpha_ord(_group.MeshObjects().size()),
_res_sum(group.MeshObjects().size(),0),
_res_to_mass(group.MeshObjects().size(),0),
_host_fs(group.MeshObjects().size(),0),
_spikeCounts(group.MeshObjects().size(), 0),
_thresholds(group.MeshObjects().size(), 0),
_resets(group.MeshObjects().size(), 0),
_reset_ws(group.MeshObjects().size(), 0),
_refractories(group.MeshObjects().size(), 0),
_blockSize(256),
_vec_num_objects(group.FiniteSizeNumObjects()),
_vec_num_object_offsets(group.FiniteSizeOffsets()),
_host_vec_objects_to_index(group._vec_objects_to_index.size(),0),
_host_vec_objects_refract_times(group._vec_objects_refract_times.size(), 0),
_host_vec_objects_refract_index(group._vec_objects_refract_index.size(), 0),
_numBlocks( (_n + _blockSize - 1) / _blockSize)
{
	this->FillMass();
    this->FillMapData();
    this->FillReversalMap(group.MeshObjects(),group.MapReversal());
	this->FillRefractoryTimes(group.Tau_ref());
    this->FillResetMap(group.MeshObjects(),group.MapReset());

	this->FillFiniteVectors();
	this->FillSpikesAndSpikeCounts();
	this->EstimateGridThresholdsResetsRefractories(group.MeshObjects(), group.MapReset(), group.Tau_ref());

}

void CudaOde2DSystemAdapter::EstimateGridThresholdsResetsRefractories(const std::vector<TwoDLib::Mesh>& vec_mesh,
	const std::vector<std::vector<TwoDLib::Redistribution> >& vec_vec_reset, const std::vector<MPILib::Time>& times) {
	for (inttype m = 0; m < _mesh_size; m++) {
		_refractories[m] = times[m];
		unsigned int middle_cell = int(vec_vec_reset[m].size() / 2.0);
		unsigned int threshold_reset_dimension = vec_mesh[m].getGridThresholdResetDirection();
		double min_threshold = (vec_mesh[m].getGridResolutionByDimension(threshold_reset_dimension) * vec_mesh[m].getGridCellWidthByDimension(threshold_reset_dimension)) 
			+ vec_mesh[m].getGridBaseByDimension(threshold_reset_dimension);
		for (unsigned int check = 0; check < vec_vec_reset[m].size(); check++) {
			if (min_threshold > vec_mesh[m].Quad(vec_vec_reset[m][check]._from[0], vec_vec_reset[m][check]._from[1]).Centroid()[0])
				min_threshold = vec_mesh[m].Quad(vec_vec_reset[m][check]._from[0], vec_vec_reset[m][check]._from[1]).Centroid()[0];
		}
		_thresholds[m] = min_threshold;
		_resets[m] = vec_mesh[m].Quad(vec_vec_reset[m][middle_cell]._to[0], vec_vec_reset[m][middle_cell]._to[1]).Centroid()[0];
		
		// To calculate the reset_w, get the w difference between from and to then add a little based on the alpha value.
		_reset_ws[m] = vec_mesh[m].Quad(vec_vec_reset[m][middle_cell]._to[0], vec_vec_reset[m][middle_cell]._to[1]).Centroid()[1] - vec_mesh[m].Quad(vec_vec_reset[m][middle_cell]._from[0], vec_vec_reset[m][middle_cell]._from[1]).Centroid()[1];
		_reset_ws[m] -= vec_mesh[m].getGridCellWidthByDimension(vec_mesh[m].getGridThresholdResetJumpDirection()) * (1.0 - vec_vec_reset[m][middle_cell]._alpha);
	}
}

CudaOde2DSystemAdapter::CudaOde2DSystemAdapter
(
    TwoDLib::Ode2DSystemGroup& group,
	unsigned int mesh_objects_start_index
): CudaOde2DSystemAdapter(group, group.MeshObjects()[0].TimeStep(), mesh_objects_start_index)
{
}

void CudaOde2DSystemAdapter::FillSpikesAndSpikeCounts() {
	checkCudaErrors(cudaMalloc((inttype**)&_spikes, _group.NumObjects() * sizeof(inttype)));
	for (inttype m = 0; m < _mesh_size; m++) {
		inttype numBlocks = (_vec_num_objects[m] + _blockSize - 1) / _blockSize;
		checkCudaErrors(cudaMalloc((inttype**)&_spikeCounts[m], numBlocks * sizeof(inttype)));
	}
}

void CudaOde2DSystemAdapter::TransferMapData()
{

	for (inttype i = 0; i < _n; i++) {
		_hostmap[i] = _group.Map(i);
		_hostunmap[_group.Map(i)] = i;	
	}

    checkCudaErrors(cudaMemcpy(_map,&_hostmap[0],_n*sizeof(inttype),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_unmap, &_hostunmap[0], _n * sizeof(inttype), cudaMemcpyHostToDevice));
}

void CudaOde2DSystemAdapter::FillRefractoryTimes(const std::vector<MPILib::Time>& times) {
	for(inttype m = 0; m < _mesh_size; m++){
		_nr_refractory_steps[m] = 2 + static_cast<int>(std::floor(times[m] / _network_time_step));
		_refractory_prop[m] = std::abs(std::fmod(times[m],_network_time_step) - _network_time_step) < 0.000001 ? 0 : std::fmod(times[m],_network_time_step)/_network_time_step;
	}
}

void CudaOde2DSystemAdapter::FillMapData(){
    checkCudaErrors(cudaMalloc((inttype**)&_map,_n*sizeof(inttype)));
	checkCudaErrors(cudaMalloc((inttype**)&_unmap, _n * sizeof(inttype)));
	checkCudaErrors(cudaMalloc((inttype**)&_map_cumulative_value, _n * sizeof(inttype)));
	checkCudaErrors(cudaMalloc((inttype**)&_map_strip_length_value, _n * sizeof(inttype)));

	for (inttype i = 0; i < _n; i++) {
		_hostmap[i] = _group.Map(i);
		_hostunmap[_group.Map(i)] = i;
	}

	checkCudaErrors(cudaMemcpy(_map, &_hostmap[0], _n * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_unmap, &_hostunmap[0], _n * sizeof(inttype), cudaMemcpyHostToDevice));

	_host_map_cumulative_value = _group.BuildMapCumulatives();
	_host_map_strip_length_value = _group.BuildMapLengths();

	checkCudaErrors(cudaMemcpy(_map_cumulative_value, &_host_map_cumulative_value[0], _n * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_map_strip_length_value, &_host_map_strip_length_value[0], _n * sizeof(inttype), cudaMemcpyHostToDevice));
}

void CudaOde2DSystemAdapter::DeleteMass()
{
    cudaFree(_mass);
}

void CudaOde2DSystemAdapter::DeleteMapData()
{
    cudaFree(_map);
	cudaFree(_unmap);
}

CudaOde2DSystemAdapter::~CudaOde2DSystemAdapter()
{
	this->DeleteFiniteVectors();
    this->DeleteMass();
    this->DeleteMapData();
    this->DeleteReversalMap();
    this->DeleteResetMap();
}

void CudaOde2DSystemAdapter::FillMass()
{
    checkCudaErrors(cudaMalloc((fptype**)&_mass,_n*sizeof(fptype)));

    for(inttype i = 0; i < _n; i++)
        _hostmass[i] = _group.Mass()[i];
    this->Validate();
     checkCudaErrors(cudaMemcpy(_mass,&_hostmass[0],_n*sizeof(fptype),cudaMemcpyHostToDevice));
}

void CudaOde2DSystemAdapter::FillFiniteVectors() {
	// Malloc the object cell index vector
	checkCudaErrors(cudaMalloc((inttype**)&_vec_objects_to_index, _group.NumObjects() * sizeof(inttype)));
	
	// Malloc the refract times
	checkCudaErrors(cudaMalloc((fptype**)&_vec_objects_refract_times, _group.NumObjects() * sizeof(fptype)));

	// Malloc the refract times
	checkCudaErrors(cudaMalloc((inttype**)&_vec_objects_refract_index, _group.NumObjects() * sizeof(inttype)));

	for (inttype i = 0; i < _group.NumObjects(); i++) {
		_host_vec_objects_to_index[i] = _group._vec_objects_to_index[i];
		_host_vec_objects_refract_times[i] = _group._vec_objects_refract_times[i];
		_host_vec_objects_refract_index[i] = _group._vec_objects_refract_index[i];
	}

	// Copy values
	checkCudaErrors(cudaMemcpy(_vec_objects_to_index, &_host_vec_objects_to_index[0], _group.NumObjects() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vec_objects_refract_times, &_host_vec_objects_refract_times[0], _group.NumObjects() * sizeof(fptype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vec_objects_refract_index, &_host_vec_objects_refract_index[0], _group.NumObjects() * sizeof(inttype), cudaMemcpyHostToDevice));
}

void CudaOde2DSystemAdapter::TransferFiniteObjects() {
	for (inttype i = 0; i < _group.NumObjects(); i++) {
		_host_vec_objects_to_index[i] = _group._vec_objects_to_index[i];
		_host_vec_objects_refract_times[i] = _group._vec_objects_refract_times[i];
	}

	checkCudaErrors(cudaMemcpy(_vec_objects_to_index, &_host_vec_objects_to_index[0], _group.NumObjects() * sizeof(inttype), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_vec_objects_refract_times, &_host_vec_objects_refract_times[0], _group.NumObjects() * sizeof(fptype), cudaMemcpyHostToDevice));
}

void CudaOde2DSystemAdapter::DeleteFiniteVectors()
{
	cudaFree(_vec_objects_to_index);
	cudaFree(_vec_objects_refract_times);
	cudaFree(_vec_objects_refract_index);
}

void CudaOde2DSystemAdapter::Validate() const
{
    // check wether the mass array of the Ode2DSystemGroup has been initialized properly. This means the mass must
    // add up to the number of meshes

    fptype sum = 0.;
    for(int i = 0; i < _n; i++)
       sum += _hostmass[i];

    fptype nmesh = static_cast<fptype>(_group.MeshObjects().size());
    if (fabs(sum - nmesh ) > tolerance){
	fprintf(stderr,"Total mass unequal to number of mesh objects:%f, %f\n",sum,nmesh);
        exit(0);
    }
}

void CudaOde2DSystemAdapter::EvolveOnDevice() {
	_group.EvolveWithoutMeshUpdate();
	inttype numBlocks = (_n + _blockSize - 1) / _blockSize;
	evolveMap << <numBlocks, _blockSize >> > (_n, 0, _map, _unmap, _map_cumulative_value, _map_strip_length_value, _group._T());
}

void CudaOde2DSystemAdapter::EvolveOnDevice(std::vector<inttype>& meshes) {
	_group.EvolveWithoutMeshUpdate();

	for (unsigned int m  : meshes) {
		unsigned int count = _offsets[m+1] - _offsets[m];

		inttype numBlocks = (count + _blockSize - 1) / _blockSize;
		evolveMap << <numBlocks, _blockSize >> > (count, _offsets[m], _map, _unmap, _map_cumulative_value, _map_strip_length_value, _group._T());
	}
}

void CudaOde2DSystemAdapter::UpdateMapData() {
	checkCudaErrors(cudaMemcpy(&_hostmap[0], _map, _n * sizeof(inttype), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&_hostunmap[0], _unmap, _n * sizeof(inttype), cudaMemcpyDeviceToHost));
	for (inttype i = 0; i < _n; i++) {
		_group.setLinearMap(i, _hostmap[i]);
		_group.setLinearUnMap(i, _hostunmap[i]);
	}
}

void CudaOde2DSystemAdapter::Evolve()
{
    _group.Evolve();
    this->TransferMapData();
}

void CudaOde2DSystemAdapter::Evolve(std::vector<inttype>& meshes)
{
    _group.Evolve(meshes);
    this->TransferMapData();
}

void CudaOde2DSystemAdapter::EvolveWithoutMeshUpdate()
{
    _group.EvolveWithoutMeshUpdate();
		this->TransferMapData();
}

void CudaOde2DSystemAdapter::EvolveWithoutTransfer()
{
	_group.UpdateMap();
}

void CudaOde2DSystemAdapter::EvolveWithoutTransfer(std::vector<inttype>& meshes)
{
	_group.UpdateMap(meshes);
}

void CudaOde2DSystemAdapter::Dump(const std::vector<std::ostream*>& vec_stream, int mode)
{
     checkCudaErrors(cudaMemcpy(&_hostmass[0],_mass,_n*sizeof(fptype),cudaMemcpyDeviceToHost));
     for(inttype i = 0; i < _n; i++)
        _group.Mass()[i] = _hostmass[i];
     _group.Dump(vec_stream, mode);
}

void CudaOde2DSystemAdapter::updateGroupMass()
{

     checkCudaErrors(cudaMemcpy(&_hostmass[0],_mass,_n*sizeof(fptype),cudaMemcpyDeviceToHost));
     for(inttype i = 0; i < _n; i++){
			 _group.Mass()[i] = _hostmass[i];
		 }
}

void CudaOde2DSystemAdapter::updateFiniteObjects()
{
	checkCudaErrors(cudaMemcpy(&_host_vec_objects_to_index[0], _vec_objects_to_index, _group.NumObjects() * sizeof(inttype), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&_host_vec_objects_refract_times[0], _vec_objects_refract_times, _group.NumObjects() * sizeof(inttype), cudaMemcpyDeviceToHost));
	for (inttype i = 0; i < _group.NumObjects(); i++) {
		_group._vec_objects_to_index[i] = _host_vec_objects_to_index[i];
		_group._vec_objects_refract_times[i] = _host_vec_objects_refract_times[i];
	}
	_group.updateVecCellsToObjects();
}

void CudaOde2DSystemAdapter::updateRefractory()
{
	for (unsigned int m = 0; m < _refractory_mass_local.size(); m++)
     checkCudaErrors(cudaMemcpy(&_refractory_mass_local[m][0],_refractory_mass[m],_nr_refractory_steps[m]*_nr_resets[m]*sizeof(fptype),cudaMemcpyDeviceToHost));
}

fptype CudaOde2DSystemAdapter::sumRefractory()
{
	fptype total = 0.0;
	for (unsigned int m = 0; m < _refractory_mass_local.size(); m++){
		for(unsigned int i=0; i<((int)_refractory_mass_local[m].size()/_nr_resets[m])-2; i++){
			for(unsigned int j=0; j<_vec_alpha_ord[m].size(); j++){
				total += _refractory_mass_local[m][i*_nr_resets[m]+j] * _vec_alpha_ord[m][j];
			}
		}
		for(unsigned int i=((int)_refractory_mass_local[m].size()/_nr_resets[m])-2; i<((int)_refractory_mass_local[m].size()/_nr_resets[m])-1; i++){
			for(unsigned int j=0; j<_vec_alpha_ord[m].size(); j++){
				total += _refractory_prop[m] * _refractory_mass_local[m][i*_nr_resets[m]+j] * _vec_alpha_ord[m][j];
			}
		}
	}

	return total;
}

MPILib::Potential CudaOde2DSystemAdapter::getAvgV(unsigned int m){
	vector<MPILib::Potential> pots = _group.AvgV();
	return pots[m];
}



const std::vector<fptype>& CudaOde2DSystemAdapter::F(unsigned int n_steps) const
{
	_host_fs.clear();
	for(inttype m = 0; m < _mesh_size; m++)
	{
		if (_vec_num_objects[m] == 0) {
			inttype numBlocks = (_nr_minimal_resets[m] + _blockSize - 1) / _blockSize;
			vector<fptype> host_sum(numBlocks, 0.);
			checkCudaErrors(cudaMemcpy(&host_sum[0], _res_sum[m], numBlocks * sizeof(fptype), cudaMemcpyDeviceToHost));
			fptype sum = 0.0;
			for (auto& rate : host_sum)
				sum += rate;
			_host_fs.push_back(sum / (_time_step * n_steps));
		}
		else {
			inttype numBlocks = (_vec_num_objects[m] + _blockSize - 1) / _blockSize;
			inttype numSumBlocks = (numBlocks + _blockSize - 1) / _blockSize;
			CudaClearSpikeCounts << <numSumBlocks, _blockSize >> > (numBlocks, _spikeCounts[m]);
			countSpikesAndClear << <numBlocks, _blockSize, _blockSize * sizeof(inttype) >> > (_vec_num_objects[m], _vec_num_object_offsets[m], _spikes, _spikeCounts[m]);

			vector<inttype> host_sum(numBlocks, 0);
			checkCudaErrors(cudaMemcpy(&host_sum[0], _spikeCounts[m], numBlocks * sizeof(inttype), cudaMemcpyDeviceToHost));
			inttype sum = 0;
			for (auto& count : host_sum)
				sum += count;
				
			_host_fs.push_back(((double)sum / (double)_vec_num_objects[m]) / (_time_step * n_steps));
		}
		
	}

  return _host_fs;
}

void CudaOde2DSystemAdapter::FillResetMap
(
    const std::vector<TwoDLib::Mesh>& vec_mesh,
    const std::vector<std::vector<TwoDLib::Redistribution> >& vec_vec_reset
)
{
    cudaMalloc(&_fs, _mesh_size*sizeof(fptype));
    std::vector<fptype> vec_rates(_mesh_size,0.);
    checkCudaErrors(cudaMemcpy(_fs,&vec_rates[0],_mesh_size*sizeof(fptype),cudaMemcpyHostToDevice));

   for(inttype m = 0; m < _mesh_size; m++)
   {
			 std::map<inttype, std::vector<std::pair<inttype,fptype>>> reset_map;
			 for(inttype i = 0; i < vec_vec_reset[m].size(); i++){
				 reset_map[_group.Map(m,vec_vec_reset[m][i]._to[0],  vec_vec_reset[m][i]._to[1])].push_back(
			 									std::pair<inttype,fptype>(_group.Map(m,vec_vec_reset[m][i]._from[0],vec_vec_reset[m][i]._from[1]),
												vec_vec_reset[m][i]._alpha));
			 }

			 _nr_minimal_resets[m] = reset_map.size();
			 _nr_resets.push_back(vec_vec_reset[m].size());
			 _refractory_mass_local[m] = std::vector<fptype>(_nr_refractory_steps[m]*vec_vec_reset[m].size());

			 checkCudaErrors(cudaMalloc((fptype**)&_refractory_mass[m], _nr_refractory_steps[m]*vec_vec_reset[m].size()*sizeof(fptype)));
			 checkCudaErrors(cudaMalloc((inttype**)&_res_to_minimal[m], _nr_minimal_resets[m]*sizeof(inttype)));
			 checkCudaErrors(cudaMalloc((inttype**)&_res_from_ordered[m], vec_vec_reset[m].size()*sizeof(inttype))); 
		     checkCudaErrors(cudaMalloc((inttype**)&_res_to_ordered[m], vec_vec_reset[m].size() * sizeof(inttype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_from_counts[m], _nr_minimal_resets[m]*sizeof(fptype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_from_offsets[m], _nr_minimal_resets[m]*sizeof(fptype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_to_mass[m],_nr_minimal_resets[m]*sizeof(fptype)));
			 inttype numBlocks = (_nr_minimal_resets[m] + _blockSize - 1)/_blockSize;
			 checkCudaErrors(cudaMalloc((fptype**)&_res_sum[m], numBlocks*sizeof(fptype)));
			 std::vector<inttype> vec_to_min;
			 std::vector<inttype> vec_from_ord;
			 std::vector<inttype> vec_to_ord;
			 std::vector<inttype> counts;
			 std::vector<inttype> offsets;

			 unsigned int offset_count = 0;
			 std::map<inttype, std::vector<std::pair<inttype,fptype>>>::iterator it;
			 for ( it = reset_map.begin(); it != reset_map.end(); it++ ){
				 vec_to_min.push_back(it->first);
				 counts.push_back(it->second.size());
				 offsets.push_back(offset_count);
				 offset_count += it->second.size();
				 for(int i=0; i<it->second.size(); i++){
					 vec_from_ord.push_back(it->second[i].first);
					 vec_to_ord.push_back(it->first);
					 _vec_alpha_ord[m].push_back(it->second[i].second);
				 }
			 }

			 checkCudaErrors(cudaMalloc((fptype**)&_res_alpha_ordered[m], _vec_alpha_ord[m].size()*sizeof(fptype)));

			 checkCudaErrors(cudaMemcpy(_res_to_minimal[m],&vec_to_min[0],vec_to_min.size()*sizeof(inttype),cudaMemcpyHostToDevice));
       		 checkCudaErrors(cudaMemcpy(_res_from_ordered[m],&vec_from_ord[0],vec_from_ord.size()*sizeof(inttype),cudaMemcpyHostToDevice));
			 checkCudaErrors(cudaMemcpy(_res_to_ordered[m], &vec_to_ord[0], vec_to_ord.size() * sizeof(inttype), cudaMemcpyHostToDevice));
       		 checkCudaErrors(cudaMemcpy(_res_alpha_ordered[m],&_vec_alpha_ord[m][0],_vec_alpha_ord[m].size()*sizeof(fptype),cudaMemcpyHostToDevice));
			 checkCudaErrors(cudaMemcpy(_res_from_counts[m],&counts[0],counts.size()*sizeof(inttype),cudaMemcpyHostToDevice));
			 checkCudaErrors(cudaMemcpy(_res_from_offsets[m],&offsets[0],offsets.size()*sizeof(inttype),cudaMemcpyHostToDevice));
	  }
}

void CudaOde2DSystemAdapter::RedistributeProbability(std::vector<inttype>& meshes)
{
	for(inttype i = 0; i < meshes.size(); i++)
	{
		inttype m = meshes[i];

		if (_group.FiniteSizeNumObjects()[m] > 0)
			continue;

		// be careful to use this block size
		inttype numBlocks = (_nr_minimal_resets[m] + _blockSize - 1)/_blockSize;
		inttype numSumBlocks = (numBlocks + _blockSize - 1)/_blockSize;
		inttype numResetBlocks = (_nr_resets[m] + _blockSize - 1)/_blockSize;

		CudaClearDerivative<<<numBlocks,_blockSize>>>(_nr_minimal_resets[m],_res_to_mass[m]);
		CudaClearDerivative<<<numSumBlocks,_blockSize>>>(numBlocks,_res_sum[m]);

		for(int t = _nr_refractory_steps[m]-2; t >= 0; t--){
			MapResetShiftRefractory<<<numResetBlocks,_blockSize>>>(_nr_resets[m],_refractory_mass[m], t*_nr_resets[m]);
		}

		MapResetToRefractory<<<numResetBlocks,_blockSize>>>(_nr_resets[m],_res_from_ordered[m], _mass, _map, _refractory_mass[m]);

		GetResetMass<<<numBlocks,_blockSize>>>(_nr_minimal_resets[m], _res_to_mass[m], _refractory_mass[m],
			_res_alpha_ordered[m], _res_from_offsets[m], _res_from_counts[m]);

		MapResetThreaded<<<numBlocks,_blockSize>>>(_nr_minimal_resets[m], _mass, _refractory_mass[m],
			(_nr_refractory_steps[m]-1)*_nr_resets[m],
			_res_to_minimal[m],_res_alpha_ordered[m], _res_from_offsets[m], _res_from_counts[m], _map, _refractory_prop[m]);

		MapResetThreaded<<<numBlocks,_blockSize>>>(_nr_minimal_resets[m], _mass, _refractory_mass[m],
			(_nr_refractory_steps[m]-2)*_nr_resets[m],
			_res_to_minimal[m],_res_alpha_ordered[m], _res_from_offsets[m], _res_from_counts[m], _map, 1.0 - _refractory_prop[m]);

		SumReset<<<numBlocks,_blockSize,_blockSize*sizeof(fptype)>>>(_nr_minimal_resets[m],_res_to_mass[m],_res_sum[m]);
	}

	cudaDeviceSynchronize();

}

void CudaOde2DSystemAdapter::RedistributeProbability()
{
	std::vector<inttype> meshes(_mesh_size);
	for(int i=0;i<_mesh_size;i++)
		if (_group.FiniteSizeNumObjects()[i] == 0)
			meshes[i] = i;

	RedistributeProbability(meshes);
}

void CudaOde2DSystemAdapter::MapFinish(std::vector<inttype>& meshes)
{
	for(inttype i = 0; i < meshes.size(); i++)
	{
		inttype m = meshes[i];

		if (_group.FiniteSizeNumObjects()[m] > 0)
			continue;
		// be careful to use this block size
		inttype numBlocks = (_nr_resets[m] + _blockSize - 1)/_blockSize;
		ResetFinishThreaded<<<numBlocks,_blockSize>>>(_nr_resets[m],_res_from_ordered[m],_mass,_map);
	}

	cudaDeviceSynchronize();

}

void CudaOde2DSystemAdapter::MapFinish()
{
	std::vector<inttype> meshes(_mesh_size);
	for(int i=0;i<_mesh_size;i++)
		meshes[i] = i;

	MapFinish(meshes);
}

void CudaOde2DSystemAdapter::FillReversalMap
(
    const std::vector<TwoDLib::Mesh>& vec_mesh,
    const std::vector<std::vector<TwoDLib::Redistribution> >& vec_vec_reversal
)
{
     _n_rev = 0;
     for(inttype m = 0; m < vec_mesh.size(); m++)
         _n_rev += vec_vec_reversal[m].size();

     cudaMallocManaged(&_rev_to,    _n_rev*sizeof(inttype));
     cudaMallocManaged(&_rev_from,  _n_rev*sizeof(inttype));
     cudaMallocManaged(&_rev_alpha, _n_rev*sizeof(fptype));

     inttype index = 0;
     for(inttype m = 0; m < vec_mesh.size(); m++){
          for( const TwoDLib::Redistribution& r: vec_vec_reversal[m] ){
              _rev_to[index]   = _group.Map(m,r._to[0],r._to[1]);
              _rev_from[index] = _group.Map(m,r._from[0],r._from[1]);
              _rev_alpha[index] = r._alpha;
              index++;
          }
     }
}

void CudaOde2DSystemAdapter::RemapReversal()
{
    MapReversal<<<1,1>>>(_n_rev, _rev_from, _rev_to, _rev_alpha, _mass, _map);
	cudaDeviceSynchronize();
}

void CudaOde2DSystemAdapter::RemapReversalFiniteObjects() {
	unsigned int num_mesh_objects = _group.NumObjects() - _vec_num_object_offsets[_mesh_objects_start_index];

	inttype numBlocks = (num_mesh_objects + _blockSize - 1) / _blockSize;
	CudaReversalFiniteObjects << <numBlocks, _blockSize >> > (num_mesh_objects, _vec_num_object_offsets[_mesh_objects_start_index], _vec_objects_to_index, _n_rev, _rev_from, _rev_to, _map);
}

void CudaOde2DSystemAdapter::RedistributeFiniteObjects(double timestep, curandState* rand_state)
{
	std::vector<inttype> meshes(_mesh_size);
	for (int i = 0; i < _mesh_size; i++)
		if (_group.FiniteSizeNumObjects()[i] > 0)
			meshes[i] = i + _mesh_objects_start_index;

	RedistributeFiniteObjects(meshes, timestep, rand_state);
}

void CudaOde2DSystemAdapter::RedistributeFiniteObjects(std::vector<inttype>& meshes, double timestep, curandState* rand_state )
{
	for (inttype m : meshes) {
		if (_group.FiniteSizeNumObjects()[m] == 0)
			continue;

		inttype numBlocks = (_vec_num_objects[m] + _blockSize - 1) / _blockSize;
		CudaResetFiniteObjects << <numBlocks, _blockSize >> > (_vec_num_objects[m], _vec_num_object_offsets[m],
			_vec_objects_to_index, _vec_objects_refract_times, _vec_objects_refract_index, 
			_refractories[m], _vec_alpha_ord[m].size(), _res_from_ordered[m], _unmap, _spikes);

		CudaCheckRefractingFiniteObjects << <numBlocks, _blockSize >> > (_vec_num_objects[m], _vec_num_object_offsets[m],
			_vec_objects_to_index, _vec_objects_refract_times, _vec_objects_refract_index, timestep,
			_vec_alpha_ord[m].size(), _res_from_ordered[m], _res_to_ordered[m], _res_alpha_ordered[m], rand_state, _map, _unmap);
	}
	
}

void CudaOde2DSystemAdapter::RedistributeGridFiniteObjects(curandState* rand_state)
{
	std::vector<inttype> meshes(_mesh_objects_start_index);
	for (int i = 0; i < _mesh_objects_start_index; i++)
		if (_group.FiniteSizeNumObjects()[i] > 0)
			meshes[i] = i;

	RedistributeGridFiniteObjects(meshes, rand_state);
}

void CudaOde2DSystemAdapter::RedistributeGridFiniteObjects(std::vector<inttype>& meshes, curandState* rand_state)
{
	for (inttype m : meshes) {
		if (_group.FiniteSizeNumObjects()[m] == 0)
			continue;

		inttype numBlocks = (_vec_num_objects[m] + _blockSize - 1) / _blockSize;
		inttype reset_dim = _group.MeshObjects()[m].getGridThresholdResetDirection();
		inttype jump_dim = _group.MeshObjects()[m].getGridThresholdResetJumpDirection();

		int threshold_col = int((_thresholds[m]-_group.MeshObjects()[m].getGridBaseByDimension(reset_dim)) / _group.MeshObjects()[m].getGridCellWidthByDimension(reset_dim));
		int reset_col = int((_resets[m] - _group.MeshObjects()[m].getGridBaseByDimension(reset_dim)) / _group.MeshObjects()[m].getGridCellWidthByDimension(reset_dim));
		int reset_w_rows = int (_reset_ws[m] / _group.MeshObjects()[m].getGridCellWidthByDimension(jump_dim));
		int res_v = _group.MeshObjects()[m].getGridResolutionByDimension(reset_dim);
		double reset_stays_probability = (_reset_ws[m] / _group.MeshObjects()[m].getGridCellWidthByDimension(jump_dim)) - reset_w_rows;
		double refractory_time = _refractories[m];
		double timestep = _group.MeshObjects()[m].TimeStep();
		
		inttype num_cells = 1;
		for (int i = 0; i < _group.MeshObjects()[m].getGridNumDimensions(); i++)
			num_cells *= _group.MeshObjects()[m].getGridResolutionByDimension(i);

		if (_group.MeshObjects()[m].stripsAreVOriented())
			CudaGridResetFiniteObjects << <numBlocks, _blockSize >> > (_vec_num_objects[m], _vec_num_object_offsets[m], _vec_objects_to_index, _vec_objects_refract_times, _vec_objects_refract_index,
				threshold_col, reset_col, reset_w_rows, res_v, reset_stays_probability, refractory_time, timestep, _spikes, _offsets[m], rand_state, num_cells);
		else
			CudaGridResetFiniteObjectsRot << <numBlocks, _blockSize >> > (_vec_num_objects[m], _vec_num_object_offsets[m], _vec_objects_to_index, _vec_objects_refract_times, _vec_objects_refract_index,
				threshold_col, reset_col, reset_w_rows, res_v, reset_stays_probability, refractory_time, timestep, _spikes, _offsets[m], rand_state, num_cells);
	}

}

void CudaOde2DSystemAdapter::DeleteResetMap()
{
    cudaFree(_fs);

    for(inttype m = 0; m < _mesh_size; m++)
    {
		cudaFree(_res_to_minimal[m]);
        cudaFree(_res_from_ordered[m]);
		cudaFree(_res_to_ordered[m]);
				cudaFree(_res_from_counts[m]);
				cudaFree(_res_alpha_ordered[m]);
        cudaFree(_res_from_offsets[m]);
				cudaFree(_res_to_mass[m]);
				cudaFree(_res_sum[m]);
    }

}

void CudaOde2DSystemAdapter::DeleteReversalMap()
{
    cudaFree(_rev_to);
    cudaFree(_rev_from);
    cudaFree(_rev_alpha);

}
