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
		MPILib::Time network_time_step
):
_group(group),
_time_step(group.MeshObjects()[0].TimeStep()),
_network_time_step(network_time_step),
_mesh_size(group.MeshObjects().size()),
_n(group.Mass().size()),
_hostmass(_n,0.),
_hostmap(_n,0.),
_offsets(group.Offsets()),
_nr_refractory_steps(group.MeshObjects().size(),0),
_refractory_prop(group.MeshObjects().size(),0),
_refractory_mass(group.MeshObjects().size(),0),
_refractory_mass_local(group.MeshObjects().size()),
_nr_minimal_resets(_group.MeshObjects().size(),0),
_res_to_minimal(_group.MeshObjects().size(),0),
_res_from_ordered(_group.MeshObjects().size(),0),
_res_alpha_ordered(_group.MeshObjects().size(),0),
_res_from_counts(_group.MeshObjects().size(),0),
_res_from_offsets(_group.MeshObjects().size(),0),
_vec_alpha_ord(),
_res_sum(group.MeshObjects().size(),0),
_res_to_mass(group.MeshObjects().size(),0),
_host_fs(group.MeshObjects().size(),0),
_blockSize(256),
_numBlocks( (_n + _blockSize - 1) / _blockSize)
{
    this->FillMass();
    this->FillMapData();
    this->FillReversalMap(group.MeshObjects(),group.MapReversal());
		this->FillRefractoryTimes(group.Tau_ref());
    this->FillResetMap(group.MeshObjects(),group.MapReset());

}

CudaOde2DSystemAdapter::CudaOde2DSystemAdapter
(
    TwoDLib::Ode2DSystemGroup& group
): CudaOde2DSystemAdapter(group, group.MeshObjects()[0].TimeStep())
{
}

void CudaOde2DSystemAdapter::TransferMapData()
{

    for( inttype i = 0; i < _n; i++)
        _hostmap[i] = _group.Map(i);

    checkCudaErrors(cudaMemcpy(_map,&_hostmap[0],_n*sizeof(inttype),cudaMemcpyHostToDevice));
}

void CudaOde2DSystemAdapter::FillRefractoryTimes(const std::vector<MPILib::Time>& times) {
	for(inttype m = 0; m < _mesh_size; m++){
		_nr_refractory_steps[m] = 2 + static_cast<int>(std::floor(times[m] / _network_time_step));
		_refractory_prop[m] = std::abs(std::fmod(times[m],_network_time_step) - _network_time_step) < 0.000001 ? 0 : std::fmod(times[m],_network_time_step)/_network_time_step;
	}
}

void CudaOde2DSystemAdapter::FillMapData(){
    checkCudaErrors(cudaMalloc((inttype**)&_map,_n*sizeof(inttype)));

    this->TransferMapData();
}

void CudaOde2DSystemAdapter::DeleteMass()
{
    cudaFree(_mass);
}

void CudaOde2DSystemAdapter::DeleteMapData()
{
    cudaFree(_map);
}

CudaOde2DSystemAdapter::~CudaOde2DSystemAdapter()
{
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
			for(unsigned int j=0; j<_vec_alpha_ord.size(); j++){
				total += _refractory_mass_local[m][i*_nr_resets[m]+j] * _vec_alpha_ord[j];
			}
		}
		for(unsigned int i=((int)_refractory_mass_local[m].size()/_nr_resets[m])-2; i<((int)_refractory_mass_local[m].size()/_nr_resets[m])-1; i++){
			for(unsigned int j=0; j<_vec_alpha_ord.size(); j++){
				total += _refractory_prop[m] * _refractory_mass_local[m][i*_nr_resets[m]+j] * _vec_alpha_ord[j];
			}
		}
	}

	return total;
}

const std::vector<fptype>& CudaOde2DSystemAdapter::F(unsigned int n_steps) const
{
	_host_fs.clear();
	for(inttype m = 0; m < _mesh_size; m++)
	{
		inttype numBlocks = (_nr_minimal_resets[m] + _blockSize - 1)/_blockSize;
		vector<fptype> host_sum(numBlocks,0.);
		checkCudaErrors(cudaMemcpy(&host_sum[0],_res_sum[m],numBlocks*sizeof(fptype),cudaMemcpyDeviceToHost));
		fptype sum = 0.0;
		for (auto& rate: host_sum)
			sum += rate;
		_host_fs.push_back(sum/(_time_step*n_steps));
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
       
			 checkCudaErrors(cudaMalloc((fptype**)&_res_from_counts[m], _nr_minimal_resets[m]*sizeof(fptype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_from_offsets[m], _nr_minimal_resets[m]*sizeof(fptype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_to_mass[m],_nr_minimal_resets[m]*sizeof(fptype)));
			 inttype numBlocks = (_nr_minimal_resets[m] + _blockSize - 1)/_blockSize;
			 checkCudaErrors(cudaMalloc((fptype**)&_res_sum[m], numBlocks*sizeof(fptype)));
			 std::vector<inttype> vec_to_min;
			 std::vector<inttype> vec_from_ord;
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
					 _vec_alpha_ord.push_back(it->second[i].second);
				 }
			 }

			checkCudaErrors(cudaMalloc((fptype**)&_res_alpha_ordered[m], _vec_alpha_ord.size()*sizeof(fptype)));

			 checkCudaErrors(cudaMemcpy(_res_to_minimal[m],&vec_to_min[0],vec_to_min.size()*sizeof(inttype),cudaMemcpyHostToDevice));
       checkCudaErrors(cudaMemcpy(_res_from_ordered[m],&vec_from_ord[0],vec_from_ord.size()*sizeof(inttype),cudaMemcpyHostToDevice));
       checkCudaErrors(cudaMemcpy(_res_alpha_ordered[m],&_vec_alpha_ord[0],_vec_alpha_ord.size()*sizeof(fptype),cudaMemcpyHostToDevice));
			 checkCudaErrors(cudaMemcpy(_res_from_counts[m],&counts[0],counts.size()*sizeof(inttype),cudaMemcpyHostToDevice));
			 checkCudaErrors(cudaMemcpy(_res_from_offsets[m],&offsets[0],offsets.size()*sizeof(inttype),cudaMemcpyHostToDevice));
	  }
}

void CudaOde2DSystemAdapter::RedistributeProbability(std::vector<inttype>& meshes)
{
	for(inttype i = 0; i < meshes.size(); i++)
  {
			inttype m = meshes[i];
			// be careful to use this block size
			inttype numBlocks = (_nr_minimal_resets[m] + _blockSize - 1)/_blockSize;
			inttype numSumBlocks = (numBlocks + _blockSize - 1)/_blockSize;
			inttype numResetBlocks = (_nr_resets[m] + _blockSize - 1)/_blockSize;

			CudaClearDerivative<<<numBlocks,_blockSize>>>(_nr_minimal_resets[m],_res_to_mass[m],_mass);
			CudaClearDerivative<<<numSumBlocks,_blockSize>>>(numBlocks,_res_sum[m],_mass);

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
		meshes[i] = i;

	RedistributeProbability(meshes);
}

void CudaOde2DSystemAdapter::MapFinish(std::vector<inttype>& meshes)
{
	for(inttype i = 0; i < meshes.size(); i++)
  {
			inttype m = meshes[i];
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



void CudaOde2DSystemAdapter::DeleteResetMap()
{
    cudaFree(_fs);

    for(inttype m = 0; m < _mesh_size; m++)
    {
				cudaFree(_res_to_minimal[m]);
        cudaFree(_res_from_ordered[m]);
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
