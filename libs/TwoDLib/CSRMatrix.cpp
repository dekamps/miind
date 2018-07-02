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
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <fstream>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include "CSRMatrix.hpp"
#include "Ode2DSystem.hpp"
#include "TwoDLibException.hpp"
using namespace TwoDLib;

void CSRMatrix::Initialize(const TransitionMatrix& mat, MPILib::Index mesh_index){

	std::vector<Redistribution> vec_dummy;

	const std::vector<TransitionMatrix::TransferLine>& matrix = mat.Matrix();

	// this must be number of cells, not size of matrix
	vector< vector<MPILib::Index> > vec_mat(_sys.Mass().size());
	vector< vector<double> > mat_vals(_sys.Mass().size());

	Validate(mat);

	// transpose
	for (auto& line: matrix){
		for (auto& r: line._vec_to_line){
			vec_mat[_sys.Map(mesh_index,r._to[0],r._to[1])].push_back(_sys.Map(mesh_index, line._from[0],line._from[1]));
			mat_vals[_sys.Map(mesh_index,r._to[0],r._to[1])].push_back(r._fraction);
		}
	}
    // csr for mat
	CSR(vec_mat,mat_vals);
}

CSRMatrix::CSRMatrix
(
	const TransitionMatrix& mat,
	const Ode2DSystemGroup& sys,
	MPILib::Index           mesh_index
):
_sys(sys),
_efficacy(mat.Efficacy()),
_val(0),
_ia(0),
_ja(0),
_mesh_index(mesh_index),
_i_offset(sys.Offsets()[mesh_index])
{
	Initialize(mat,mesh_index);
}

void CSRMatrix::Validate(const TransitionMatrix& mat){
	// At this stage, the _coordinates list, which has been constructed from the TransitionMatrix
	// should match one one one the cells that are in the Ode2DSystem. A failure is likely to
	// originate from not inserting stationary bins into the Mesh object after it has been read from
	// file. The Stat file used to construct the TransitionMatrix contains these stationary bins

	const Mesh& m = _sys.MeshObjects()[_mesh_index];

	const std::vector<TransitionMatrix::TransferLine>& lines = mat.Matrix();

	MPILib::Number count = 0;
	// check for consistency in strip 0; if not the same number of stationary points, there
	// is an inconsistency between the config file of the Mesh and that of the TransitionMatrix

	for (auto& line: lines)
		if (line._from[0] == 0) count++;

	if ( count != m.NrCellsInStrip(0) )	
	   throw TwoDLib::TwoDLibException("There is a stationary point in your mesh file, but no entries in the mat file that lead away from it.");
      
}

void CSRMatrix::CSR(const vector<vector<MPILib::Index> >& vec_mat, const vector<vector<double> >& mat_vals){
	_ia.push_back(0);

	for (MPILib::Index i = 0; i < vec_mat.size(); i++){
		_ia.push_back( _ia.back() + vec_mat[i].size());
		for (MPILib::Index j = 0; j < vec_mat[i].size(); j++){
			_val.push_back(mat_vals[i][j]);
			_ja.push_back(vec_mat[i][j]);
		}
	}
}

void CSRMatrix::MV(vector<double>& out, const vector<double>& in){

	assert( out.size() + 1 == _ia.size());

	MPILib::Index nr_rows = _ia.size() - 1;
	for (MPILib::Index i = 0; i < nr_rows; i++){
	  for(MPILib::Index j = _ia[i]; j < _ia[i+1]; j++){
			out[i] += _val[j]*in[_ja[j]];
		}
	}
}


void CSRMatrix::MVMapped
(
	vector<double>&       dydt,
	const vector<double>& vec_mass,
	double                rate
) const
{
	unsigned int nr_rows = _ia.size() - 1;

#pragma omp parallel for 

	for (MPILib::Index i = 0; i < nr_rows; i++){
	  MPILib::Index i_r =_sys.Map(i+_i_offset);
	  for( MPILib::Index j = _ia[i]; j < _ia[i+1]; j++){
			 int j_m = _sys.Map(_ja[j]+_i_offset);
			 dydt[i_r] += rate*_val[j]*vec_mass[j_m];
		 }
	         dydt[i_r] -= rate*vec_mass[i_r];
	}
}
