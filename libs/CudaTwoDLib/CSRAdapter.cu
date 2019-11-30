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
#include "CSRAdapter.cuh"

using namespace CudaTwoDLib;

const fptype TOLERANCE = 1e-9;


#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

void CSRAdapter::FillMatrixMaps(const std::vector<TwoDLib::CSRMatrix>& vecmat)
{
  unsigned int vecmat_index = 0;
   for(inttype m = 0; m < _vecmats.size()+_grid_transforms.size(); m++)
   {
        // Grid transitions sit in between grid transforms and mesh transitions and aren't in vecmat as we're building them later.
       if(m >= _grid_transforms.size() && m < (_vecmats.size()+_grid_transforms.size()) - (vecmat.size()-_grid_transforms.size()))
          continue;


       _offsets[m] = vecmat[vecmat_index].Offset();
       _nr_rows[m] = vecmat[vecmat_index].NrRows();
      
       _nval[m] = vecmat[vecmat_index].Val().size();
       checkCudaErrors(cudaMalloc((fptype**)&_val[m],_nval[m]*sizeof(fptype)));
       // dont't depend on Val() being of fptype
       std::vector<fptype> vecval;
       for (fptype val: vecmat[vecmat_index].Val())
           vecval.push_back(val);
       checkCudaErrors(cudaMemcpy(_val[m],&vecval[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

       _nia[m] = vecmat[vecmat_index].Ia().size();
       checkCudaErrors(cudaMalloc((inttype**)&_ia[m],_nia[m]*sizeof(inttype)));
       std::vector<inttype> vecia;
       for(inttype ia: vecmat[vecmat_index].Ia())
           vecia.push_back(ia);
       checkCudaErrors(cudaMemcpy(_ia[m],&vecia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));


       _nja[m] = vecmat[vecmat_index].Ja().size();
       checkCudaErrors(cudaMalloc((inttype**)&_ja[m],_nja[m]*sizeof(inttype)));
       std::vector<inttype> vecja;
       for(inttype ja: vecmat[vecmat_index].Ja())
           vecja.push_back(ja);
       checkCudaErrors(cudaMemcpy(_ja[m],&vecja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));

       vecmat_index++;
   }
}

void CSRAdapter::InitializeStaticGridEfficacies(const std::vector<inttype>& vecindex,const std::vector<fptype>& efficacy) {

  for(inttype e = 0; e < efficacy.size(); e++)
  {
    inttype m = e + _grid_transforms.size();

    _offsets[m] = _group.getGroup().Offsets()[vecindex[e]];
    _nr_rows[m] = _nr_rows[vecindex[e]];

    _nval[m] = _nr_rows[vecindex[e]] * 2; // each cell has two transition values
    checkCudaErrors(cudaMalloc((fptype**)&_val[m],_nval[m]*sizeof(fptype)));

    _nia[m] = _nr_rows[vecindex[e]]; // each cell has one row in the transition matrix
    checkCudaErrors(cudaMalloc((inttype**)&_ia[m],_nia[m]*sizeof(inttype)));

    _nja[m] = _nr_rows[vecindex[e]] * 2; // each cell has transitions from two other cells
    checkCudaErrors(cudaMalloc((inttype**)&_ja[m],_nja[m]*sizeof(inttype)));

    inttype numBlocks = (_nr_rows[vecindex[e]] + _blockSize - 1)/_blockSize;

    CudaCalculateGridEfficacies<<<numBlocks,_blockSize>>>(_nr_rows[vecindex[e]],
      efficacy[e], _cell_widths[vecindex[e]],
      _val[m], _ia[m], _ja[m]);
  }    

}

void CSRAdapter::InitializeStaticGridEfficacySlow(const inttype vecindex, const inttype connindex, const fptype efficacy) {

  inttype m = connindex + _grid_transforms.size();

  _offsets[m] = _group.getGroup().Offsets()[vecindex];
  _nr_rows[m] = _nr_rows[vecindex];

  // This is going to be slow : we have to generate the forward transitions before we
  // can translate to val, ia and ja. We have to do this because we no longer know
  // how many incoming cells there will be (when not v dependent it's always two incoming cells)
  std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex]);
  std::vector<std::vector<double>> vals(_nr_rows[vecindex]);
  for (unsigned int i=0; i<_nr_rows[vecindex]; i++) {
    fptype eff = efficacy;
    inttype ofs = (inttype)abs(eff / _cell_widths[vecindex]);
    fptype g = (fptype)fabs(eff / _cell_widths[vecindex]) - ofs;
    fptype s = 1.0 - g;

    int o1 = efficacy > 0 ? ofs : -ofs;
    int o2 = efficacy > 0 ? (ofs+1) : (ofs-1);

    int r1 = (i+o1)%_nr_rows[vecindex];
    unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex] : r1;

    int r2 = (i+o2)%_nr_rows[vecindex];
    unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex] : r2;

    inds[ind_1].push_back(i);
    inds[ind_2].push_back(i);
    vals[ind_1].push_back(g);
    vals[ind_2].push_back(s);
  }

  std::vector<inttype> ia;
  std::vector<inttype> ja;
  std::vector<fptype> val;
  ia.push_back(0);

  for (MPILib::Index i = 0; i < inds.size(); i++){
    ia.push_back( ia.back() + inds[i].size());
    for (MPILib::Index j = 0; j < inds[i].size(); j++){
      val.push_back((fptype)vals[i][j]);
      ja.push_back(inds[i][j]);
    }
  }

  _nval[m] = val.size();
  checkCudaErrors(cudaMalloc((fptype**)&_val[m],_nval[m]*sizeof(fptype)));
  checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

  _nia[m] = ia.size();
  checkCudaErrors(cudaMalloc((inttype**)&_ia[m],_nia[m]*sizeof(inttype)));
  checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

  _nja[m] = ja.size();
  checkCudaErrors(cudaMalloc((inttype**)&_ja[m],_nja[m]*sizeof(inttype)));
  checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
}

void CSRAdapter::UpdateGridEfficacySlow(const inttype vecindex, const inttype connindex, const fptype efficacy) {

  inttype m = connindex + _grid_transforms.size();

  // This is going to be slow : we have to generate the forward transitions before we
  // can translate to val, ia and ja. We have to do this because we no longer know
  // how many incoming cells there will be (when not v dependent it's always two incoming cells)
  std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex]);
  std::vector<std::vector<double>> vals(_nr_rows[vecindex]);
  for (unsigned int i=0; i<_nr_rows[vecindex]; i++) {
    fptype eff = efficacy;
    inttype ofs = (inttype)abs(eff / _cell_widths[vecindex]);
    fptype g = (fptype)fabs(eff / _cell_widths[vecindex]) - ofs;
    fptype s = 1.0 - g;

    int o1 = efficacy > 0 ? ofs : -ofs;
    int o2 = efficacy > 0 ? (ofs+1) : (ofs-1);

    int r1 = (i+o1)%_nr_rows[vecindex];
    unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex] : r1;

    int r2 = (i+o2)%_nr_rows[vecindex];
    unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex] : r2;

    inds[ind_1].push_back(i);
    inds[ind_2].push_back(i);
    vals[ind_1].push_back(g);
    vals[ind_2].push_back(s);
  }

  std::vector<inttype> ia;
  std::vector<inttype> ja;
  std::vector<fptype> val;
  ia.push_back(0);

  for (MPILib::Index i = 0; i < inds.size(); i++){
    ia.push_back( ia.back() + inds[i].size());
    for (MPILib::Index j = 0; j < inds[i].size(); j++){
      val.push_back((fptype)vals[i][j]);
      ja.push_back(inds[i][j]);
    }
  }

  _nval[m] = val.size();
  checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

  _nia[m] = ia.size();
  checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

  _nja[m] = ja.size();
  checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
}

void CSRAdapter::InitializeStaticGridEfficacySlowLateralEpileptor(const inttype vecindex, const inttype connindex, const fptype efficacy, const fptype tau, const fptype K, const fptype v_in) {

  inttype m = connindex + _grid_transforms.size();

  unsigned int strip_length = _group.getGroup().MeshObjects()[vecindex].NrCellsInStrip(0);

  _offsets[m] = _group.getGroup().Offsets()[vecindex];
  _nr_rows[m] = _nr_rows[vecindex];

  // This is going to be slow : we have to generate the forward transitions before we
  // can translate to val, ia and ja. We have to do this because we no longer know
  // how many incoming cells there will be (when not v dependent it's always two incoming cells)
  std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex]);
  std::vector<std::vector<double>> vals(_nr_rows[vecindex]);
  for (unsigned int i=0; i<_nr_rows[vecindex]; i++) {
    fptype eff = efficacy * (1.0 / tau);
    inttype ofs = (inttype)abs(eff / _cell_heights[vecindex]);
    fptype g = (fptype)fabs(eff / _cell_heights[vecindex]) - ofs;
    fptype s = 1.0 - g;

    int o1 = eff > 0 ? ofs : -ofs;
    int o2 = eff > 0 ? (ofs+1) : (ofs-1);

    int r1 = (i+(o1*strip_length))%_nr_rows[vecindex];
    unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex] : r1;

    int r2 = (i+(o2*strip_length))%_nr_rows[vecindex];
    unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex] : r2;

    inds[ind_1].push_back(i);
    inds[ind_2].push_back(i);
    vals[ind_1].push_back(s);
    vals[ind_2].push_back(g);
  }

  std::vector<inttype> ia;
  std::vector<inttype> ja;
  std::vector<fptype> val;
  ia.push_back(0);

  for (MPILib::Index i = 0; i < inds.size(); i++){
    ia.push_back( ia.back() + inds[i].size());
    for (MPILib::Index j = 0; j < inds[i].size(); j++){
      val.push_back((fptype)vals[i][j]);
      ja.push_back(inds[i][j]);
    }
  }

  _nval[m] = val.size();
  checkCudaErrors(cudaMalloc((fptype**)&_val[m],_nval[m]*sizeof(fptype)));
  checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

  _nia[m] = ia.size();
  checkCudaErrors(cudaMalloc((inttype**)&_ia[m],_nia[m]*sizeof(inttype)));
  checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

  _nja[m] = ja.size();
  checkCudaErrors(cudaMalloc((inttype**)&_ja[m],_nja[m]*sizeof(inttype)));
  checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
}

void CSRAdapter::UpdateGridEfficacySlowLateralEpileptor(const inttype vecindex, const inttype connindex, const fptype efficacy, const fptype tau, const fptype K, const fptype v_in) {

  inttype m = connindex + _grid_transforms.size();

  unsigned int strip_length = _group.getGroup().MeshObjects()[vecindex].NrCellsInStrip(0);

  // This is going to be slow : we have to generate the forward transitions before we
  // can translate to val, ia and ja. We have to do this because we no longer know
  // how many incoming cells there will be (when not v dependent it's always two incoming cells)
  std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex]);
  std::vector<std::vector<double>> vals(_nr_rows[vecindex]);
  for (unsigned int i=0; i<_nr_rows[vecindex]; i++) {
    fptype eff = -1.0 * (1.0 / tau) * (K * (v_in - _group.getGroup().Vs()[_offsets[vecindex]+i]));
    inttype ofs = (inttype)abs(eff / _cell_heights[vecindex]);
    fptype g = (fptype)fabs(eff / _cell_heights[vecindex]) - ofs;
    fptype s = 1.0 - g;

    int o1 = eff > 0 ? ofs : -ofs;
    int o2 = eff > 0 ? (ofs+1) : (ofs-1);

    int r1 = (i+(o1*strip_length))%_nr_rows[vecindex];
    unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex] : r1;

    int r2 = (i+(o2*strip_length))%_nr_rows[vecindex];
    unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex] : r2;

    inds[ind_1].push_back(i);
    inds[ind_2].push_back(i);
    vals[ind_1].push_back(s);
    vals[ind_2].push_back(g);
  }

  std::vector<inttype> ia;
  std::vector<inttype> ja;
  std::vector<fptype> val;
  ia.push_back(0);

  for (MPILib::Index i = 0; i < inds.size(); i++){
    ia.push_back( ia.back() + inds[i].size());
    for (MPILib::Index j = 0; j < inds[i].size(); j++){
      val.push_back((fptype)vals[i][j]);
      ja.push_back(inds[i][j]);
    }
  }

  _nval[m] = val.size();
  checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

  _nia[m] = ia.size();
  checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

  _nja[m] = ja.size();
  checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
}

void CSRAdapter::InitializeStaticGridEfficacySlowLateral(const inttype vecindex, const inttype connindex, const fptype efficacy) {

  inttype m = connindex + _grid_transforms.size();

  unsigned int strip_length = _group.getGroup().MeshObjects()[vecindex].NrCellsInStrip(0);

  _offsets[m] = _group.getGroup().Offsets()[vecindex];
  _nr_rows[m] = _nr_rows[vecindex];

  // This is going to be slow : we have to generate the forward transitions before we
  // can translate to val, ia and ja. We have to do this because we no longer know
  // how many incoming cells there will be (when not v dependent it's always two incoming cells)
  std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex]);
  std::vector<std::vector<double>> vals(_nr_rows[vecindex]);
  for (unsigned int i=0; i<_nr_rows[vecindex]; i++) {
    fptype eff = efficacy;
    inttype ofs = (inttype)abs(eff / _cell_heights[vecindex]);
    fptype g = (fptype)fabs(eff / _cell_heights[vecindex]) - ofs;
    fptype s = 1.0 - g;

    int o1 = efficacy > 0 ? ofs : -ofs;
    int o2 = efficacy > 0 ? (ofs+1) : (ofs-1);

    int r1 = (i+(o1*strip_length))%_nr_rows[vecindex];
    unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex] : r1;

    int r2 = (i+(o2*strip_length))%_nr_rows[vecindex];
    unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex] : r2;

    inds[ind_1].push_back(i);
    inds[ind_2].push_back(i);
    vals[ind_1].push_back(s);
    vals[ind_2].push_back(g);
  }

  std::vector<inttype> ia;
  std::vector<inttype> ja;
  std::vector<fptype> val;
  ia.push_back(0);

  for (MPILib::Index i = 0; i < inds.size(); i++){
    ia.push_back( ia.back() + inds[i].size());
    for (MPILib::Index j = 0; j < inds[i].size(); j++){
      val.push_back((fptype)vals[i][j]);
      ja.push_back(inds[i][j]);
    }
  }

  _nval[m] = val.size();
  checkCudaErrors(cudaMalloc((fptype**)&_val[m],_nval[m]*sizeof(fptype)));
  checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

  _nia[m] = ia.size();
  checkCudaErrors(cudaMalloc((inttype**)&_ia[m],_nia[m]*sizeof(inttype)));
  checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

  _nja[m] = ja.size();
  checkCudaErrors(cudaMalloc((inttype**)&_ja[m],_nja[m]*sizeof(inttype)));
  checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
}

void CSRAdapter::UpdateGridEfficacySlowLateral(const inttype vecindex, const inttype connindex, const fptype efficacy) {

  inttype m = connindex + _grid_transforms.size();

  unsigned int strip_length = _group.getGroup().MeshObjects()[vecindex].NrCellsInStrip(0);

  // This is going to be slow : we have to generate the forward transitions before we
  // can translate to val, ia and ja. We have to do this because we no longer know
  // how many incoming cells there will be (when not v dependent it's always two incoming cells)
  std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex]);
  std::vector<std::vector<double>> vals(_nr_rows[vecindex]);
  for (unsigned int i=0; i<_nr_rows[vecindex]; i++) {
    fptype eff = efficacy;
    inttype ofs = (inttype)abs(eff / _cell_heights[vecindex]);
    fptype g = (fptype)fabs(eff / _cell_heights[vecindex]) - ofs;
    fptype s = 1.0 - g;

    int o1 = efficacy > 0 ? ofs : -ofs;
    int o2 = efficacy > 0 ? (ofs+1) : (ofs-1);

    int r1 = (i+(o1*strip_length))%_nr_rows[vecindex];
    unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex] : r1;

    int r2 = (i+(o2*strip_length))%_nr_rows[vecindex];
    unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex] : r2;

    inds[ind_1].push_back(i);
    inds[ind_2].push_back(i);
    vals[ind_1].push_back(s);
    vals[ind_2].push_back(g);
  }

  std::vector<inttype> ia;
  std::vector<inttype> ja;
  std::vector<fptype> val;
  ia.push_back(0);

  for (MPILib::Index i = 0; i < inds.size(); i++){
    ia.push_back( ia.back() + inds[i].size());
    for (MPILib::Index j = 0; j < inds[i].size(); j++){
      val.push_back((fptype)vals[i][j]);
      ja.push_back(inds[i][j]);
    }
  }

  _nval[m] = val.size();
  checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

  _nia[m] = ia.size();
  checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

  _nja[m] = ja.size();
  checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
}

void CSRAdapter::UpdateGridEfficacies(const std::vector<inttype>& vecindex,const std::vector<fptype>& efficacy) {
  for(inttype e = 0; e < efficacy.size(); e++)
  {
    inttype m = e + _grid_transforms.size();

    inttype numBlocks = (_nr_rows[vecindex[e]] + _blockSize - 1)/_blockSize;

    CudaCalculateGridEfficacies<<<numBlocks,_blockSize>>>(_nr_rows[vecindex[e]],
      efficacy[e], _cell_widths[vecindex[e]],
      _val[m], _ia[m], _ja[m]);
  }    
}

// Experimental : untested.
void CSRAdapter::InitializeStaticGridVDependentEfficacies(const std::vector<inttype>& vecindex,
  const std::vector<fptype>& efficacy, const std::vector<fptype>& rest_vs) {

  for(inttype e = 0; e < efficacy.size(); e++)
  {
    inttype m = e + _grid_transforms.size();

    _offsets[m] = _group.getGroup().Offsets()[vecindex[e]];
    _nr_rows[m] = _nr_rows[vecindex[e]];

    // This is going to be slow : we have to generate the forward transitions before we
    // can translate to val, ia and ja. We have to do this because we no longer know
    // how many incoming cells there will be (when not v dependent it's always two incoming cells)
    std::vector<std::vector<unsigned int>> inds(_nr_rows[vecindex[m]]);
    std::vector<std::vector<double>> vals(_nr_rows[vecindex[m]]);
    for (unsigned int i=0; i<_nr_rows[vecindex[m]]; i++) {
      fptype eff = efficacy[m] * (_group.getGroup().Vs()[_offsets[vecindex[m]]+i] - rest_vs[m]);
      inttype ofs = (inttype)abs(eff / _cell_widths[vecindex[m]]);
      fptype g = (fptype)fabs(eff / _cell_widths[vecindex[m]]) - ofs;
      fptype s = 1.0 - g;

      int o1 = efficacy[m] > 0 ? ofs : -ofs;
      int o2 = efficacy[m] > 0 ? (ofs+1) : (ofs-1);

      int r1 = (i-o1)%_nr_rows[vecindex[m]];
      unsigned int ind_1 = r1< 0 ? r1 + _nr_rows[vecindex[m]] : r1;

      int r2 = (i-o2)%_nr_rows[vecindex[m]];
      unsigned int ind_2 = r2< 0 ? r2 + _nr_rows[vecindex[m]] : r2;

      inds[ind_1].push_back(i);
      inds[ind_2].push_back(i);
      vals[ind_1].push_back(s);
      vals[ind_2].push_back(g);
    }

    std::vector<inttype> ia;
    std::vector<inttype> ja;
    std::vector<fptype> val;
    ia.push_back(0);

    for (MPILib::Index i = 0; i < inds.size(); i++){
      ia.push_back( ia.back() + inds[i].size());
      for (MPILib::Index j = 0; j < inds[i].size(); j++){
        val.push_back((fptype)vals[i][j]);
        ja.push_back(inds[i][j]);
      }
    }

    _nval[m] = val.size();
    checkCudaErrors(cudaMalloc((fptype**)&_val[m],_nval[m]*sizeof(fptype)));
    checkCudaErrors(cudaMemcpy(_val[m],&val[0],sizeof(fptype)*_nval[m],cudaMemcpyHostToDevice));

    _nia[m] = ia.size();
    checkCudaErrors(cudaMalloc((inttype**)&_ia[m],_nia[m]*sizeof(inttype)));
    checkCudaErrors(cudaMemcpy(_ia[m],&ia[0],sizeof(inttype)*_nia[m],cudaMemcpyHostToDevice));

    _nja[m] = ja.size();
    checkCudaErrors(cudaMalloc((inttype**)&_ja[m],_nja[m]*sizeof(inttype)));
    checkCudaErrors(cudaMemcpy(_ja[m],&ja[0],sizeof(inttype)*_nja[m],cudaMemcpyHostToDevice));
  }  
}


void CSRAdapter::DeleteMatrixMaps()
{
    for(inttype m = 0; m < _nr_m; m++)
    {
        cudaFree(_val[m]);
        cudaFree(_ia[m]);
        cudaFree(_ja[m]);
    }
}

inttype CSRAdapter::NumberIterations(const CudaOde2DSystemAdapter& group, fptype euler_timestep) const
{
    fptype tstep = group._group.MeshObjects()[0].TimeStep();
    for ( const auto& mesh: group._group.MeshObjects() )
        if (fabs(tstep - mesh.TimeStep()) > TOLERANCE){
           std::cerr << "Not all meshes in this group have the same time step. " <<  tstep << " " << mesh.TimeStep() << " " << tstep - mesh.TimeStep()  << std::endl;
           exit(0);
        }
    inttype  n_steps = static_cast<inttype>(std::round(tstep/euler_timestep));

    return n_steps;
}

void CSRAdapter::InspectMass(inttype i)
{
    std::vector<fptype> hostvec(_group._n);
    checkCudaErrors(cudaMemcpy(&hostvec[0],_group._mass,sizeof(fptype)*_group._n,cudaMemcpyDeviceToHost));
}

CSRAdapter::CSRAdapter(CudaOde2DSystemAdapter& group, const std::vector<TwoDLib::CSRMatrix>& vecmat,
 inttype nr_connections, fptype euler_timestep,
 const std::vector<inttype>& vecmat_indexes,const std::vector<inttype>& grid_transforms):
_group(group),
_euler_timestep(euler_timestep),
_nr_iterations(NumberIterations(group,euler_timestep)),
_nr_m(vecmat_indexes.size()+grid_transforms.size()),
_nr_streams(vecmat_indexes.size()+grid_transforms.size()),
_vecmats(vecmat_indexes),
_grid_transforms(grid_transforms),
_cell_widths(CellWidths(vecmat)),
_cell_heights(CellHeights(vecmat)),
_nval(std::vector<inttype>(vecmat_indexes.size()+grid_transforms.size())),
_val(std::vector<fptype*>(vecmat_indexes.size()+grid_transforms.size())),
_nia(std::vector<inttype>(vecmat_indexes.size()+grid_transforms.size())),
_ia(std::vector<inttype*>(vecmat_indexes.size()+grid_transforms.size())),
_nja(std::vector<inttype>(vecmat_indexes.size()+grid_transforms.size())),
_ja(std::vector<inttype*>(vecmat_indexes.size()+grid_transforms.size())),
_offsets(vecmat_indexes.size()+grid_transforms.size()),
_nr_rows(vecmat_indexes.size()+grid_transforms.size()),
_blockSize(256),
_numBlocks( (_group._n + _blockSize - 1) / _blockSize)
{
    this->FillMatrixMaps(vecmat);
    this->FillDerivative();
    this->CreateStreams();
}

CSRAdapter::CSRAdapter(CudaOde2DSystemAdapter& group, const std::vector<TwoDLib::CSRMatrix>& vecmat, fptype euler_timestep):
CSRAdapter(group,vecmat,vecmat.size(),euler_timestep,
std::vector<inttype>(),std::vector<inttype>()){
  for(unsigned int i=0; i<vecmat.size(); i++)
   _vecmats.push_back(i);
}

CSRAdapter::~CSRAdapter()
{
    this->DeleteMatrixMaps();
    this->DeleteDerivative();
    this->DeleteStreams();
}

void CSRAdapter::CreateStreams()
{
    _streams = (cudaStream_t *)malloc(_nr_streams*sizeof(cudaStream_t));
    for(int i = 0; i < _nr_streams; i++)
       cudaStreamCreate(&_streams[i]);
}

void CSRAdapter::DeleteStreams()
{
   free(_streams);
}

void CSRAdapter::FillDerivative()
{
    checkCudaErrors(cudaMalloc((fptype**)&_dydt,_group._n*sizeof(fptype)));
}

void CSRAdapter::DeleteDerivative()
{
    cudaFree(_dydt);
}

void CSRAdapter::ClearDerivative()
{
  inttype n=_group._n;
  CudaClearDerivative<<<_numBlocks,_blockSize>>>(n,_dydt,_group._mass);
}

std::vector<fptype> CSRAdapter::CellWidths(const std::vector<TwoDLib::CSRMatrix>& vecmat) const
{
	std::vector<fptype> vecret;
	for (inttype m = 0; m < _grid_transforms.size(); m++){
    vecret.push_back(_group.getGroup().MeshObjects()[_grid_transforms[m]].getCellWidth());
  }
	return vecret;
}

std::vector<fptype> CSRAdapter::CellHeights(const std::vector<TwoDLib::CSRMatrix>& vecmat) const
{
	std::vector<fptype> vecret;
	for (inttype m = 0; m < _grid_transforms.size(); m++){
    vecret.push_back(_group.getGroup().MeshObjects()[_grid_transforms[m]].getCellHeight());
  }
	return vecret;
}

void CSRAdapter::CalculateDerivative(const std::vector<fptype>& vecrates)
{
    for(inttype m : _vecmats)
    {
        // be careful to use this block size
        inttype numBlocks = (_nr_rows[m] + _blockSize - 1)/_blockSize;
        CudaCalculateDerivative<<<numBlocks,_blockSize>>>(_nr_rows[m],vecrates[m],_dydt,_group._mass,_val[m],_ia[m],_ja[m],_group._map,_offsets[m]);
    }

}

void CSRAdapter::CalculateMeshGridDerivative(const std::vector<inttype>& vecindex, const std::vector<fptype>& vecrates)
{
  for(int n=0; n<vecrates.size(); n++)
  {
    inttype mat_index = _grid_transforms.size() + n;
    // be careful to use this block size
    inttype numBlocks = (_nr_rows[mat_index] + _blockSize - 1)/_blockSize;
    CudaCalculateDerivative<<<numBlocks,_blockSize,0,_streams[vecindex[n]]>>>(_nr_rows[mat_index],vecrates[n],_dydt,_group._mass,_val[mat_index],_ia[mat_index],_ja[mat_index],_group._map,_offsets[mat_index]);
  }

  cudaDeviceSynchronize();

}

void CSRAdapter::SingleTransformStep()
{
  for(inttype m : _grid_transforms)
  {
      // be careful to use this block size
      inttype numBlocks = (_nr_rows[m] + _blockSize - 1)/_blockSize;
      CudaSingleTransformStep<<<numBlocks,_blockSize,0,_streams[m]>>>(_nr_rows[m],_dydt,_group._mass,_val[m],_ia[m],_ja[m],_group._map,_offsets[m]);
  }
}

void CSRAdapter::AddDerivative()
{
  EulerStep<<<_numBlocks,_blockSize>>>(_group._n,_dydt,_group._mass,_euler_timestep);
}

void CSRAdapter::AddDerivativeFull()
{
  EulerStep<<<_numBlocks,_blockSize>>>(_group._n,_dydt,_group._mass, 1.0);
}
