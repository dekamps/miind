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
    TwoDLib::Ode2DSystemGroup& group
):
_group(group),
_time_step(group.MeshObjects()[0].TimeStep()),
_mesh_size(group.MeshObjects().size()),
_n(group.Mass().size()),
_hostmass(_n,0.),
_hostmap(_n,0.),
_offsets(group.Offsets()),
_res_to(group.MeshObjects().size(),0),
_res_from(group.MeshObjects().size(),0),
_res_alpha(group.MeshObjects().size(),0),
_res_sum(group.MeshObjects().size(),0),
_res_to_mass(group.MeshObjects().size(),0),
_nr_to_cells(group.MeshObjects().size(),0),
_host_fs(group.MeshObjects().size(),0),
_reset_nval(std::vector<inttype>(group.MeshObjects().size())),
_reset_val(std::vector<fptype*>(group.MeshObjects().size())),
_reset_nia(std::vector<inttype>(group.MeshObjects().size())),
_reset_ia(std::vector<inttype*>(group.MeshObjects().size())),
_reset_nja(std::vector<inttype>(group.MeshObjects().size())),
_reset_ja(std::vector<inttype*>(group.MeshObjects().size())),
_blockSize(256),
_numBlocks( (_n + _blockSize - 1) / _blockSize)
{
    this->FillMass();
		this->FillDerivative();
    this->FillMapData();
    this->FillReversalMap(group.MeshObjects(),group.MapReversal());
    this->FillResetMap(group.MeshObjects(),group.MapReset());
		this->FillResetMatrixMaps(group.ResetCSR());
}

void CudaOde2DSystemAdapter::FillResetMatrixMaps(const std::vector<TwoDLib::CSRMatrix>& vecmat)
{
   for(inttype m = 0; m < vecmat.size(); m++)
   {
       _reset_nval[m] = vecmat[m].Val().size();
       checkCudaErrors(cudaMalloc((fptype**)&_reset_val[m],_reset_nval[m]*sizeof(fptype)));
       // dont't depend on Val() being of fptype
       std::vector<fptype> vecval;
       for (fptype val: vecmat[m].Val())
           vecval.push_back(val);
       checkCudaErrors(cudaMemcpy(_reset_val[m],&vecval[0],sizeof(fptype)*_reset_nval[m],cudaMemcpyHostToDevice));

       _reset_nia[m] = vecmat[m].Ia().size();
       checkCudaErrors(cudaMalloc((inttype**)&_reset_ia[m],_reset_nia[m]*sizeof(inttype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_to_mass[m],60000*sizeof(fptype)));
       std::vector<inttype> vecia;
       for(inttype ia: vecmat[m].Ia())
           vecia.push_back(ia);
       checkCudaErrors(cudaMemcpy(_reset_ia[m],&vecia[0],sizeof(inttype)*_reset_nia[m],cudaMemcpyHostToDevice));

       _reset_nja[m] = vecmat[m].Ja().size();
       checkCudaErrors(cudaMalloc((inttype**)&_reset_ja[m],_reset_nja[m]*sizeof(inttype)));
       std::vector<inttype> vecja;
       for(inttype ja: vecmat[m].Ja())
           vecja.push_back(ja);
       checkCudaErrors(cudaMemcpy(_reset_ja[m],&vecja[0],sizeof(inttype)*_reset_nja[m],cudaMemcpyHostToDevice));
   }
}

void CudaOde2DSystemAdapter::DeleteCSR()
{
    for(inttype m = 0; m < _mesh_size; m++)
    {
        cudaFree(_reset_val[m]);
        cudaFree(_reset_ia[m]);
        cudaFree(_reset_ja[m]);
    }
}

void CudaOde2DSystemAdapter::TransferMapData()
{

    for( inttype i = 0; i < _n; i++)
        _hostmap[i] = _group.Map(i);

    checkCudaErrors(cudaMemcpy(_map,&_hostmap[0],_n*sizeof(inttype),cudaMemcpyHostToDevice));
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

void CudaOde2DSystemAdapter::FillDerivative()
{
    checkCudaErrors(cudaMalloc((fptype**)&_dydt,_n*sizeof(fptype)));
}

void CudaOde2DSystemAdapter::DeleteDerivative()
{
    cudaFree(_dydt);
}

void CudaOde2DSystemAdapter::ClearDerivative()
{
  CudaClearDerivative<<<_numBlocks,_blockSize>>>(_n,_dydt,_mass);
}

void CudaOde2DSystemAdapter::AddDerivativeFull()
{
  EulerStep<<<_numBlocks,_blockSize>>>(_n,_dydt,_mass, 1.0);
}

CudaOde2DSystemAdapter::~CudaOde2DSystemAdapter()
{
    this->DeleteMass();
		this->DeleteDerivative();
    this->DeleteMapData();
    this->DeleteReversalMap();
		this->DeleteCSR();
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
	fprintf(stderr,"Total mass  unequal to number of mesh objects:%f, %f\n",sum,nmesh);
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
     for(inttype i = 0; i < _n; i++)
        _group.Mass()[i] = _hostmass[i];
}

const std::vector<fptype>& CudaOde2DSystemAdapter::F() const
{
	_host_fs.clear();
	for(inttype m = 0; m < _mesh_size; m++)
	{
		inttype numBlocks = (60000 + 256 - 1)/256;
		vector<fptype> host_sum(numBlocks,0.);
		checkCudaErrors(cudaMemcpy(&host_sum[0],_res_sum[m],numBlocks*sizeof(fptype),cudaMemcpyDeviceToHost));
		fptype sum = 0.0;
		for (auto& rate: host_sum)
				sum += rate;
		_host_fs.push_back(sum*0.7/_time_step);
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
       _nr_resets.push_back(vec_vec_reset[m].size());
       checkCudaErrors(cudaMalloc((inttype**)&_res_to[m],   vec_vec_reset[m].size()*sizeof(inttype)));
       checkCudaErrors(cudaMalloc((inttype**)&_res_from[m], vec_vec_reset[m].size()*sizeof(inttype)));
       checkCudaErrors(cudaMalloc((fptype**)&_res_alpha[m], vec_vec_reset[m].size()*sizeof(fptype)));
			 checkCudaErrors(cudaMalloc((fptype**)&_res_sum[m], ((60000 + 256 - 1)/256)*sizeof(fptype)));
       std::vector<inttype> vec_to;
       std::vector<inttype> vec_from;
       std::vector<fptype>  vec_alpha;
       for(inttype i = 0; i < vec_vec_reset[m].size(); i++)
       {
           vec_to.push_back(_group.Map(m,vec_vec_reset[m][i]._to[0],  vec_vec_reset[m][i]._to[1]));
           vec_from.push_back(_group.Map(m,vec_vec_reset[m][i]._from[0],vec_vec_reset[m][i]._from[1]));
           vec_alpha.push_back(vec_vec_reset[m][i]._alpha);
       }
       checkCudaErrors(cudaMemcpy(_res_to[m],&vec_to[0],vec_to.size()*sizeof(inttype),cudaMemcpyHostToDevice));
       checkCudaErrors(cudaMemcpy(_res_from[m],&vec_from[0],vec_from.size()*sizeof(inttype),cudaMemcpyHostToDevice));
       checkCudaErrors(cudaMemcpy(_res_alpha[m],&vec_alpha[0],vec_alpha.size()*sizeof(fptype),cudaMemcpyHostToDevice));
  }
}

void CudaOde2DSystemAdapter::RedistributeProbability()
{
    // for (inttype m = 0; m < _mesh_size; m++){
    //     fptype* f = _fs+m;
    //     MapReset<<<1,1>>>(_nr_resets[m],_res_from[m],_res_to[m],_res_alpha[m],_mass,_map,f);
    // }
}

void CudaOde2DSystemAdapter::RedistributeProbabilityThreaded()
{
	for(inttype m = 0; m < _mesh_size; m++)
	{
			// be careful to use this block size
			inttype numBlocks = (60000 + 256 - 1)/256;
			inttype numSumBlocks = (numBlocks + 256 - 1)/256;

			CudaClearDerivative<<<numBlocks,256>>>(60000,_res_to_mass[m],_mass);
			CudaClearDerivative<<<numSumBlocks,256>>>(numBlocks,_res_sum[m],_mass);

			MapResetThreaded<<<numBlocks,256>>>(60000, _res_to_mass[m],_dydt,_mass,_reset_val[m],_reset_ia[m],_reset_ja[m],_map,_offsets[m]);

			SumReset<<<numBlocks,256,256*sizeof(fptype)>>>(60000,_res_to_mass[m],_res_sum[m]);
	}

	// F();
	// for(int i=0; i<_host_fs.size(); i++)
	// 	std::cout << _host_fs[i] << "\n";
}

void CudaOde2DSystemAdapter::MapFinish()
{
    for (inttype m = 0; m < _mesh_size; m++)
        ResetFinish<<<1,1>>>(_nr_resets[m],_res_from[m],_mass,_map);
}

void CudaOde2DSystemAdapter::MapFinishThreaded()
{
	for(inttype m = 0; m < _mesh_size; m++)
  {
      // be careful to use this block size
      inttype numBlocks = (_nr_resets[m] + 256 - 1)/256;
			ResetFinishThreaded<<<numBlocks,256>>>(_nr_resets[m],_res_from[m],_mass,_map);
  }
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
}


void CudaOde2DSystemAdapter::DeleteResetMap()
{
    cudaFree(_fs);

    for(inttype m = 0; m < _mesh_size; m++)
    {
	cudaFree(_res_to[m]);
        cudaFree(_res_from[m]);
        cudaFree(_res_alpha[m]);
				cudaFree(_res_sum[m]);
    }

}

void CudaOde2DSystemAdapter::DeleteReversalMap()
{
    cudaFree(_rev_to);
    cudaFree(_rev_from);
    cudaFree(_rev_alpha);

}
