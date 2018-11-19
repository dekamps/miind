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
#include "Ode2DSystem.hpp"
#include "TwoDLibException.hpp"


using namespace TwoDLib;

Ode2DSystem::Ode2DSystem
(
	const Mesh& m,
	const vector<Redistribution>&  vec_reversal,
	const vector<Redistribution>&  vec_reset,
	const MPILib::Time& tau_refractive
):
_mesh(m),
_vec_length(InitializeLength(m)),
_vec_cumulative(InitializeCumulative(m)),
_vec_mass(InitializeMass()),
_vec_area(InitializeArea(m)),
_it(0),
_f(0),
_map(InitializeMap()),
_vec_reversal(vec_reversal),
_vec_reset(vec_reset),
_reversal(*this,_vec_mass),
_reset(*this,_vec_mass),
_reset_refractive(*this,_vec_mass,_it,tau_refractive,_vec_reset),
_clean(*this,_vec_mass),
_tau_refractive(tau_refractive)
{
	assert(m.TimeStep() != 0.0);
	this->CheckConsistency();
}

bool Ode2DSystem::CheckConsistency() const {

	std::ostringstream ost_err;
	ost_err << "Mesh inconsistent with mapping: ";
	for (const Redistribution& r: _vec_reversal){
		if ( r._from[0] >= _mesh.NrStrips() ){
			ost_err << "reversal. NrStrips: " << _mesh.NrStrips() << ", from: " << r._from[0];
			throw TwoDLib::TwoDLibException(ost_err.str());
		}
		if ( r._from[1] >= _mesh.NrCellsInStrip(r._from[0] ) ){
			ost_err << "reversal. Nr cells in strip from: " <<  _mesh.NrCellsInStrip(r._from[0] ) << ",from: " << r._from[0] << "\n";
			ost_err << "In total there are: " << _mesh.NrStrips() << " strips." << std::endl;
			throw TwoDLib::TwoDLibException(ost_err.str());
		}
	}

	for (const Redistribution& r: _vec_reset){
		if ( r._from[0] >= _mesh.NrStrips() ){
			ost_err << "reset. NrStrips: " << _mesh.NrStrips() << ", from: " << r._from[0] << std::endl;
			throw TwoDLib::TwoDLibException(ost_err.str());
		}
		if ( r._from[1] >= _mesh.NrCellsInStrip(r._from[0] ) ){
			ost_err << "reset. Nr cells in strip r._from[0]: " <<  _mesh.NrCellsInStrip(r._from[0] ) << ", from: " << r._from[1];
			throw TwoDLib::TwoDLibException(ost_err.str());
		}
	}

	return true;
}

vector<MPILib::Index> Ode2DSystem::InitializeCumulative(const Mesh& m) const
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

vector<double> Ode2DSystem::InitializeMass() const
{
	return vector<double>(_vec_cumulative.back(),0.0);
}

vector<unsigned int> Ode2DSystem::InitializeLength(const Mesh& m) const
{
	vector<unsigned int> vec_ret;
	for(unsigned int i = 0; i < m.NrStrips();i++)
		vec_ret.push_back(m.NrCellsInStrip(i));
	return vec_ret;
}

vector<double> Ode2DSystem::InitializeArea(const Mesh& m) const
{
	vector<double> vec_ret;
	for (unsigned int i = 0; i < m.NrStrips(); i++)
		for (unsigned int j = 0; j < m.NrCellsInStrip(i); j++ )
			vec_ret.push_back(m.Quad(i,j).SignedArea());
	return vec_ret;
}
void Ode2DSystem::Initialize(unsigned int i, unsigned int j){
	_vec_mass[this->Map(i,j)] = 1.0;
}
std::vector< std::vector<MPILib::Index> > Ode2DSystem::InitializeMap() const
{
	std::vector<std::vector<MPILib::Index> > vec_map;
	MPILib::Index count = 0;
	for (MPILib::Index i = 0; i < _mesh.NrStrips();i++){
		std::vector<MPILib::Index> vec_strip;
		for (MPILib::Index j = 0; j < _mesh.NrCellsInStrip(i); j++){
			vec_strip.push_back(count++);
		}
		vec_map.push_back(vec_strip);
	}
	return vec_map;
}

void Ode2DSystem::Dump(std::ostream& ost, int mode) const
{
	ost.precision(10);
	if (mode == 0) {
	for (unsigned int i = 0; i < _mesh.NrStrips(); i++)
		for (unsigned int j = 0; j < _mesh.NrCellsInStrip(i); j++ )
			// a division by _vec_area[this->Map(i,j)] is wrong
			// the fabs is required since we don't care about the sign of the area and
			// must write out a positive density
			ost << i << "\t" << j << "\t" << " " << fabs(_vec_mass[this->Map(i,j)]/_mesh.Quad(i,j).SignedArea()) << "\t";
	} else {
		for (unsigned int i = 0; i < _mesh.NrStrips(); i++)
			for (unsigned int j = 0; j < _mesh.NrCellsInStrip(i); j++ )
				ost << i << "\t" << j << "\t" << " " << _vec_mass[this->Map(i,j)] << "\t";
		}

}

void Ode2DSystem::Evolve()
{
	_it += 1;
	_f = 0;
	UpdateMap();
}

void Ode2DSystem::EvolveWithoutMeshUpdate(){
	_it += 1;
	_f = 0;
}

void Ode2DSystem::UpdateMap()
{
	for (MPILib::Index i = 1; i < _mesh.NrStrips(); i++){
		// yes! i = 1. strip 0 is not supposed to have dynamics
		for (MPILib::Index j = 0; j < _mesh.NrCellsInStrip(i); j++ ){
			_map[i][j] =_vec_cumulative[i] + modulo(j-_it,_vec_length[i]);
		}
	}
}

void Ode2DSystem::RemapReversal(){
	std::for_each(_vec_reversal.begin(),_vec_reversal.end(),_reversal);
}

void Ode2DSystem::RedistributeProbability()
{
	RedistributeProbability(1);
}

void Ode2DSystem::RedistributeProbability(MPILib::Number steps)
{
	if (_tau_refractive == 0.)
		for(auto& m: _vec_reset)
			_reset(m);
	else
		for(auto& m: _vec_reset)
			_reset_refractive(m);

	std::for_each(_vec_reset.begin(),_vec_reset.end(),_clean);

	_f /= (_mesh.TimeStep()*steps);
}

double Ode2DSystem::AvgV() const
{
	double av = 0.;
	for(MPILib::Index i = 0; i < _mesh.NrStrips(); i++){
		for(MPILib::Index j = 0; j < _mesh.NrCellsInStrip(i); j++){
			double V = _mesh.Quad(i,j).Centroid()[0];
			av += V*_vec_mass[this->Map(i,j)];
		}
	}

	return av;
}
