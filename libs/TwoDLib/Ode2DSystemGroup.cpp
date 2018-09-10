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
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Ode2DSystemGroup.hpp"
#include "TwoDLibException.hpp"


using namespace TwoDLib;


std::vector<Ode2DSystemGroup::Clean> Ode2DSystemGroup::InitializeClean()
{
	std::vector<Ode2DSystemGroup::Clean> vec_ret;
	for (MPILib::Index m = 0; m < _mesh_list.size(); m++){
		Clean clean(*this,_vec_mass,m);
		vec_ret.push_back(clean);
	}
	return vec_ret;
}

std::vector<Ode2DSystemGroup::Reset> Ode2DSystemGroup::InitializeReset()
{
	std::vector<Ode2DSystemGroup::Reset> vec_ret;
	for (MPILib::Index m = 0; m < _mesh_list.size(); m++){
		Reset reset(*this,_vec_mass,m);
		vec_ret.push_back(reset);
	}
	return vec_ret;
}

std::vector<Ode2DSystemGroup::Reversal> Ode2DSystemGroup::InitializeReversal()
{
	std::vector<Ode2DSystemGroup::Reversal> vec_ret;
	for (MPILib::Index m = 0; m < _mesh_list.size(); m++){
		Reversal reversal(*this,_vec_mass,m);
		vec_ret.push_back(reversal);
	}
	return vec_ret;
}

Ode2DSystemGroup::Ode2DSystemGroup
(
	const std::vector<Mesh>& mesh_list,
	const std::vector<std::vector<Redistribution> >& vec_reversal,
	const std::vector<std::vector<Redistribution> >& vec_reset
):
_mesh_list(mesh_list),
_vec_mesh_offset(MeshOffset(mesh_list)),
_vec_length(InitializeLengths(mesh_list)),
_vec_cumulative(InitializeCumulatives(mesh_list)),
_vec_mass(InitializeMass()),
_vec_area(InitializeArea(mesh_list)),
_t(0),
_fs(std::vector<MPILib::Rate>(mesh_list.size(),0.0)),
_avs(std::vector<MPILib::Potential>(mesh_list.size(),0.0)),
_map(InitializeMap()),
_linear_map(InitializeLinearMap()),
_vec_reversal(vec_reversal),
_vec_reset(vec_reset),
_reversal(InitializeReversal()),
_reset(InitializeReset()),
_clean(InitializeClean())
{
	for(const auto& m: _mesh_list)
		assert(m.TimeStep() != 0.0);
	this->CheckConsistency();
}

std::vector<MPILib::Number> Ode2DSystemGroup::ã(const std::vector<Mesh>& l) const
{
	std::vector<MPILib::Number> vec_ret{0}; // first offset is 0
	for (const Mesh& m: l){
		MPILib::Number n_cell = 0;
		for (MPILib::Index i = 0; i < m.NrStrips(); i++)
			for( MPILib::Index j = 0; j < m.NrCellsInStrip(i); j++)
				n_cell++;
		vec_ret.push_back(n_cell + vec_ret.back());
	}

	return vec_ret;
}


std::vector<MPILib::Index> Ode2DSystemGroup::InitializeCumulative(const Mesh& m) const
{
	unsigned int sum = 0;
	vector<unsigned int> vec_ret;
	vec_ret.push_back(0);
	for(unsigned int i = 0; i < m.NrStrips(); i++){
		sum += m.NrCellsInStrip(i);
		vec_ret.push_back(sum);
	}
	return vec_ret;
}

std::vector<std::vector<MPILib::Index> > Ode2DSystemGroup::InitializeCumulatives(const std::vector<Mesh>& mesh_list)
{
	std::vector<std::vector<MPILib::Index> > vec_ret;
	for(const Mesh& m: mesh_list)
		vec_ret.push_back(this->InitializeCumulative(m));
	return vec_ret;
}

vector<MPILib::Potential> Ode2DSystemGroup::InitializeMass() const
{
	MPILib::Number n_cells = 0;
	for(const std::vector<MPILib::Index>& v: _vec_cumulative)
		n_cells += v.back();

	return vector<MPILib::Potential>(n_cells,0.0);
}

std::vector<MPILib::Index> Ode2DSystemGroup::InitializeLength(const Mesh& m) const
{
	std::vector<MPILib::Index> vec_ret;
	for(unsigned int i = 0; i < m.NrStrips();i++)
		vec_ret.push_back(m.NrCellsInStrip(i));
	return vec_ret;
}

std::vector< std::vector<MPILib::Index> > Ode2DSystemGroup::InitializeLengths(const std::vector<Mesh>& list)
{
	std::vector< std::vector<MPILib::Index> > vec_ret;
	for(const Mesh& m: list)
		vec_ret.push_back(this->InitializeLength(m));

	return vec_ret;
}


vector<MPILib::Potential> Ode2DSystemGroup::InitializeArea(const std::vector<Mesh>& vec) const
{
	vector<MPILib::Potential> vec_ret;
	for (const Mesh& m: _mesh_list){
		for (unsigned int i = 0; i < m.NrStrips(); i++)
			for (unsigned int j = 0; j < m.NrCellsInStrip(i); j++ )
				vec_ret.push_back(m.Quad(i,j).SignedArea());
	}
	return vec_ret;
}

void Ode2DSystemGroup::Initialize(MPILib::Index m, MPILib::Index i, MPILib::Index j){
	_vec_mass[this->Map(m,i,j)] = 1.0;
}

std::vector< std::vector< std::vector<MPILib::Index> > > Ode2DSystemGroup::InitializeMap() const
{
	std::vector< std::vector<std::vector<MPILib::Index> > > vec_map;
	MPILib::Index count = 0;

	for(const Mesh& mesh: _mesh_list){
		std::vector<std::vector<MPILib::Index> > vec_mesh;
		for (MPILib::Index i = 0; i < mesh.NrStrips();i++){
			std::vector<MPILib::Index> vec_strip;
			for (MPILib::Index j = 0; j < mesh.NrCellsInStrip(i); j++){
				vec_strip.push_back(count++);
			}
			vec_mesh.push_back(vec_strip);
		}
		vec_map.push_back(vec_mesh);
	}
	return vec_map;
}

std::vector<MPILib::Index> Ode2DSystemGroup::InitializeLinearMap()
{
	std::vector<MPILib::Index> vec_ret;
	MPILib::Index counter = 0;
	for( const Mesh& mesh: _mesh_list){
		for (MPILib::Index i = 0; i < mesh.NrStrips(); i++)
			for (MPILib::Index j = 0; j < mesh.NrCellsInStrip(i); j++)
				vec_ret.push_back(counter++);
	}
	return vec_ret;
}

void Ode2DSystemGroup::Dump(const std::vector<std::ostream*>& vecost, int mode) const
{
	assert(vecost.size() == _mesh_list.size());
	for(MPILib::Index m = 0; m < _mesh_list.size(); m++){
		vecost[m]->precision(10);
		if (mode == 0) {
			for (unsigned int i = 0; i < _mesh_list[m].NrStrips(); i++)
				for (unsigned int j = 0; j < _mesh_list[m].NrCellsInStrip(i); j++ )
					// a division by _vec_area[this->Map(i,j)] is wrong
					// the fabs is required since we don't care about the sign of the area and
					// must write out a positive density
					(*vecost[m]) << i << "\t" << j << "\t" << " " << fabs(_vec_mass[this->Map(m,i,j)]/_mesh_list[m].Quad(i,j).SignedArea()) << "\t";
		} else {
			for (unsigned int i = 0; i < _mesh_list[m].NrStrips(); i++)
				for (unsigned int j = 0; j < _mesh_list[m].NrCellsInStrip(i); j++ )
					(*vecost[m]) << i << "\t" << j << "\t" << " " << _vec_mass[this->Map(m,i,j)] << "\t";
		}
	}
}

void Ode2DSystemGroup::Evolve()
{
	_t += 1;
	for (MPILib::Rate& f: _fs)
		f = 0.;
	this->UpdateMap();
}

void Ode2DSystemGroup::UpdateMap()
{
	MPILib::Index counter = 0;
	for (MPILib::Index m = 0; m < _mesh_list.size(); m++){ // we need the index for mapping, so no range-based loop
		for(MPILib::Index i_stat = 0; i_stat < _mesh_list[m].NrCellsInStrip(0); i_stat++)
			_linear_map[counter++] = i_stat + _vec_mesh_offset[m]; // the stationary strip needs to be handled separately
		for (MPILib::Index i = 1; i < _mesh_list[m].NrStrips(); i++){
			// yes! i = 1. strip 0 is not supposed to have dynamics
			for (MPILib::Index j = 0; j < _mesh_list[m].NrCellsInStrip(i); j++ ){
				MPILib::Index ind = _vec_cumulative[m][i] + modulo(j-_t,_vec_length[m][i]) + _vec_mesh_offset[m];
				_map[m][i][j] = ind;
				_linear_map[counter++] = ind;
			}
		}
	}
}

void Ode2DSystemGroup::RemapReversal(){

	for(MPILib::Index m = 0; m < _mesh_list.size(); m++)
		std::for_each(_vec_reversal[m].begin(),_vec_reversal[m].end(),_reversal[m]);
}

void Ode2DSystemGroup::RedistributeProbability()
{
	for(MPILib::Index m=0; m < _mesh_list.size(); m++){
		std::for_each(_vec_reset[m].begin(),_vec_reset[m].end(),_reset[m]);
		std::for_each(_vec_reset[m].begin(),_vec_reset[m].end(),_clean[m]);
	}
	MPILib::Time t_step = _mesh_list[0].TimeStep(); // they all should have the same time step
	for (MPILib::Rate& f: _fs)
		f /= t_step;
}

const std::vector<MPILib::Potential>& Ode2DSystemGroup::AvgV() const
{
	// Rate calculation for non-threshold crossing models such as Fitzhugh-Nagumo
	for(MPILib::Index m = 0; m < _mesh_list.size(); m++){
		MPILib::Potential av = 0.;
		for(MPILib::Index i = 0; i < _mesh_list[m].NrStrips(); i++){
			for(MPILib::Index j = 0; j < _mesh_list[m].NrCellsInStrip(i); j++){
				MPILib::Potential V = _mesh_list[m].Quad(i,j).Centroid()[0];
				av += V*_vec_mass[this->Map(m,i,j)];
			}
		}
		const_cast<MPILib::Potential&>(_avs[m]) = av;
	}
	return _avs;
}

bool Ode2DSystemGroup::CheckConsistency() const {

	std::ostringstream ost_err;
	ost_err << "Mesh inconsistent with mapping: ";
	// it is allowed to have no reversal and reset mappings
	if (_vec_reversal.size() == 0 && _vec_reset.size() == 0)
		return true;
	else // but if you have them, they must match the mesh list
	{
		if (_vec_reset.size() != _mesh_list.size() ){
			ost_err << "Reset mapping vector size does not match mesh list size";
			return false;
		}
		if (_vec_reversal.size() != _mesh_list.size() ){
			ost_err << "Reversal mapping vector size does not match mesh list size";
			return false;
		}
	}

	for (MPILib::Index m = 0; m < _mesh_list.size(); m++ ){
		for (const Redistribution& r: _vec_reversal[m]){
			if ( r._from[0] >= _mesh_list[m].NrStrips() ){
				ost_err << "reversal. NrStrips: " << _mesh_list[m].NrStrips() << ", from: " << r._from[0];
				throw TwoDLib::TwoDLibException(ost_err.str());
			}
			if ( r._from[1] >= _mesh_list[m].NrCellsInStrip(r._from[0] ) ){
				ost_err << "reversal. Nr cells in strip from: " <<  _mesh_list[m].NrCellsInStrip(r._from[0] ) << ",from: " << r._from[0] << "\n";
				ost_err << "In total there are: " << _mesh_list[m].NrStrips() << " strips." << std::endl;
				throw TwoDLib::TwoDLibException(ost_err.str());
			}
		}

		for (const Redistribution& r: _vec_reset[m]){
			if ( r._from[0] >= _mesh_list[m].NrStrips() ){
				ost_err << "reset. NrStrips: " << _mesh_list[m].NrStrips() << ", from: " << r._from[0] << std::endl;
				throw TwoDLib::TwoDLibException(ost_err.str());
			}
			if ( r._from[1] >= _mesh_list[m].NrCellsInStrip(r._from[0] ) ){
				ost_err << "reset. Nr cells in strip r._from[0]: " <<  _mesh_list[m].NrCellsInStrip(r._from[0] ) << ", from: " << r._from[1];
				throw TwoDLib::TwoDLibException(ost_err.str());
			}
		}
	}
	return true;
}
