// Copyright (c) 2005 - 2011 Marc de Kamps
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
#include <cmath>
#include "Hexagon.h"
using namespace std;
namespace {
	
        const MPILib::Number N_Hexagon   = 6;
	const float  Phi_hexagon = 2*3.1415926535F/6;
	const float  TOLERANCE   = 0.01F;
	const float  EPS         = 1e-4F;

}

bool IsPointInGrid(const std::vector<IdGrid>& vec_grid, const IdGrid& point){
        for (MPILib::Index i = 0; i < vec_grid.size(); i++){
		float sqr = pow(vec_grid[i]._x - point._x,2) + pow(vec_grid[i]._y - point._y,2);
		if ( sqrt(sqr) < TOLERANCE)
			return true;
	}
	return false;
}

void BuildHexagonalGrid
(	vector<IdGrid>*					pvec_grid, 
	vector<pair<MPILib::NodeId,MPILib::NodeId> >*	pvec_link, 
	MPILib::Number n_rings
)
{
	int id_count = 0;
	vector<IdGrid> vec_seed;
	
	IdGrid seed = {MPILib::NodeId(id_count++), 0.0, 0.0};
	vec_seed.push_back(seed);
	pvec_grid->push_back(seed);
	while(n_rings-- > 0){
		// add a ring around all seeds, unless the corresponding point is already in the list, the nodes
		// that are added in this pass are the seed for the enxt round
		vector<IdGrid> vec_new_seed;
		for( MPILib::Index i_seed = 0; i_seed < vec_seed.size(); i_seed++){
		  for (MPILib::Index i_hex = 0; i_hex < N_Hexagon; i_hex++){
				IdGrid new_point;
				new_point._x = vec_seed[i_seed]._x + cos(i_hex*Phi_hexagon);
				new_point._y = vec_seed[i_seed]._y + sin(i_hex*Phi_hexagon);
				if (! IsPointInGrid(vec_new_seed,new_point) && !IsPointInGrid(*pvec_grid,new_point)){
				        new_point._id = MPILib::NodeId(id_count++);
					vec_new_seed.push_back(new_point);
				}
			}
		}
		vec_seed = vec_new_seed;
		for (MPILib::Index i = 0; i < vec_seed.size(); i++)
			pvec_grid->push_back(vec_seed[i]);
	}
	vector<IdGrid>& vec_grid = *pvec_grid;
	// generate neighbourpairs
	for (MPILib::Index i = 0; i < vec_grid.size(); i++)
	      for (MPILib::Index j = 0; j < i; j++)
			if ( fabs(pow(vec_grid[i]._x -vec_grid[j]._x,2) + pow(vec_grid[i]._y - vec_grid[j]._y,2) - 1.0) < EPS){
			        pair<MPILib::NodeId,MPILib::NodeId> p;
				p.first  = vec_grid[i]._id;
				p.second = vec_grid[j]._id;
				pvec_link->push_back(p);
			}
				
}

MPILib::Number NumberOfNeighbours(const vector<IdGrid>& vec_grid, MPILib::NodeId id){
	// first walk through the grid to establish the maximal radius
	// all points not in the outer ring have 6 neighbours.
	// all other points four neighbours, unless phi is i*2pi/6, which have three neighbours
	float max_rad = 0;
	for (MPILib::Index i = 0; i < vec_grid.size(); i++){
		float rad = sqrt(pow(vec_grid[i]._x,2) + pow(vec_grid[i]._y,2));
		if (rad > max_rad)
			max_rad = rad;
	}
	MPILib::Index j = 0;
	for(MPILib::Index i = 0; i < vec_grid.size(); i++ )
		if (id == vec_grid[i]._id){
			j = i;
			break;
		}

	bool b_outer_ring =
		fabs(  vec_grid[j]._y - max_rad*sqrt(3.0)*0.5 )    < EPS ||
		fabs( -vec_grid[j]._y - max_rad*sqrt(3.0)*0.5 )    < EPS ||
		fabs(  sqrt(3.0)/3.0*vec_grid[j]._y - vec_grid[j]._x - max_rad) < EPS ||
		fabs(  sqrt(3.0)/3.0*vec_grid[j]._y - vec_grid[j]._x + max_rad) < EPS ||
		fabs( -sqrt(3.0)/3.0*vec_grid[j]._y - vec_grid[j]._x - max_rad) < EPS ||
		fabs( -sqrt(3.0)/3.0*vec_grid[j]._y - vec_grid[j]._x + max_rad) < EPS;

	if ( ! b_outer_ring)
		return N_Hexagon;
	else {
		// outer ring
		float phi = atan(vec_grid[j]._y/vec_grid[j]._x);
		// should be close to a multiple of (-) Phi_hexagon
		float f = phi/Phi_hexagon;
		// so either the floor or the ceil of f should be close to f
		if (fabs(floor(f) -f) < EPS || fabs(ceil(f) -f) < EPS)
			return 3;
		else
			return 4;
	}
}

vector<MPILib::NodeId> NodesOntoThisNode(const vector<nodepair>& vec_pair, MPILib::NodeId id)
{
        vector<MPILib::NodeId> vec_ret;
	for (MPILib::Index i = 0; i < vec_pair.size(); i++){
		if (vec_pair[i].first == id)
			vec_ret.push_back(vec_pair[i].second);
		if (vec_pair[i].second == id)
			vec_ret.push_back(vec_pair[i].first);
	}
	return vec_ret;
}


void WriteGridToStream
(
	const vector<IdGrid>& vec_grid, 
	const vector<pair<MPILib::NodeId, MPILib::NodeId> >& 
	vec_link, ostream& s
){
        for (MPILib::Index i = 0; i < vec_grid.size(); i++)
		s << vec_grid[i]._id << "\t" 
		  << vec_grid[i]._x  << "\t" 
		  << vec_grid[i]._y  << "\t" 
		  << NumberOfNeighbours(vec_grid,vec_grid[i]._id) << "\n";

	s << "-\n";
	for (MPILib::Index j = 0; j < vec_link.size(); j++)
		s << vec_link[j].first << " " << vec_link[j].second << "\n";
}


/*
void PrintGrid(const vector<IdGrid>& vec_grid){
        for(MPILib::Index i = 0; i < vec_grid.size(); i++){
		cout << "---" << endl;
		cout << vec_grid[i]._id << endl;
		cout << vec_grid[i]._x  << endl;
		cout << vec_grid[i]._y  << endl;
	}
}
*/
