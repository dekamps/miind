// Copyright (c) 2005 - 2016 Marc de Kamps
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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include "MPILib/include/TypeDefinitions.hpp"
#include "CorrectStrays.hpp"
#include "TwoDLibException.hpp"

namespace {
	MPILib::Index find_nearest
	(
		const std::vector<TwoDLib::Coordinates>& vec_th,
		const TwoDLib::Hit& h,
		const TwoDLib::Mesh& m
	){
		vector<double> dist;

		TwoDLib::Point ph = m.Quad(h._cell[0],h._cell[1]).Centroid();
		for(const auto& c: vec_th){
			TwoDLib::Point pc = m.Quad(c[0],c[1]).Centroid();
			double d = std::pow(ph[0] -  pc[0],2) + std::pow(ph[1] - pc[1],2);
			dist.push_back(d);
		}

		ptrdiff_t i_min = std::distance(dist.begin(), std::min_element(dist.begin(),dist.end()));
		return i_min;
	}
}

TwoDLib::TransitionList TwoDLib::CorrectStrays
(
	const TwoDLib::TransitionList& l,               //! list of all hits for a transition
	const vector<TwoDLib::Coordinates>& ths,     	//! list of cells on threshold
	const vector<TwoDLib::Coordinates>& above,	    //! list of cells above threshold
	const Mesh& m 									//! need to be able to find the positions of coordinates
){
	// the transition list contains transitions, regardless of whether some of them may be above
	// threshold. CorrectStrays maps a transition to a cell above back to a cell on threshold. The cleaned
	// translation list is returned.

	TransitionList list_ret;
	list_ret._origin = l._origin;
	list_ret._number = l._number;


	std::vector<TwoDLib::Hit> vec_above;

	// we make a list of destinations above threshold; they will have to be remapped onto the threshold
	for(const TwoDLib::Hit& h: l._destination_list){
		if ( std::find(above.begin(), above.end(), h._cell) != above.end() ){
			vec_above.push_back(h);
		}	else {
			list_ret._destination_list.push_back(h);
		}
	}

	vector<MPILib::Index> vec_close;
	// for each destination in the above list we must find the threshold cell that is closest
	for (const auto& h: vec_above){
		MPILib::Index i_n = find_nearest(ths,h, m);
		vec_close.push_back(i_n);
	}

	// now for each above cell check if there is already a hit corresponding to that threshold cell in the destination list
	for(MPILib::Index i = 0; i < vec_above.size(); i++ ){
		Coordinates nearest_th = ths[vec_close[i]];

		// if there is, increase the count by the hit of the above cell
		// if there isn't insert a hit with a count of the above cell

		bool b_hit = false;
		MPILib::Index i_list = 0;
		for(MPILib::Index i = 0; i < list_ret._destination_list.size(); i++){
			const Hit& h = list_ret._destination_list[i];
			if (h._cell[0] == nearest_th[0] && h._cell[1] == nearest_th[1]){
				b_hit = true;
				i_list = i;
			}
		}
		if (b_hit){
			list_ret._destination_list[i_list]._count += vec_above[i]._count;
		} else {
			// if there isn't insert a hit with a count of the above cell
			Hit new_hit;
			new_hit._cell = nearest_th;
			new_hit._count = vec_above[i]._count;
			list_ret._destination_list.push_back(new_hit);
		}

	}

	return list_ret;
}

TwoDLib::TransitionList TwoDLib::CorrectStraysProportion
(
	const TwoDLib::TransitionList& l,               //! list of all hits for a transition
	const vector<TwoDLib::Coordinates>& ths,     	//! list of cells on threshold
	const vector<TwoDLib::Coordinates>& above,	    //! list of cells above threshold
	const Mesh& m 									//! need to be able to find the positions of coordinates
){
	// the transition list contains transitions, regardless of whether some of them may be above
	// threshold. CorrectStrays maps a transition to a cell above back to a cell on threshold. The cleaned
	// translation list is returned.

	TransitionList list_ret;
	list_ret._origin = l._origin;
	list_ret._number = l._number;


	std::vector<TwoDLib::Hit> vec_above;

	// we make a list of destinations above threshold; they will have to be remapped onto the threshold
	for(const TwoDLib::Hit& h: l._destination_list){
		if ( std::find(above.begin(), above.end(), h._cell) != above.end() ){
			vec_above.push_back(h);
		}	else {
			list_ret._destination_list.push_back(h);
		}
	}

	vector<MPILib::Index> vec_close;
	// for each destination in the above list we must find the threshold cell that is closest
	for (const auto& h: vec_above){
		MPILib::Index i_n = find_nearest(ths,h, m);
		vec_close.push_back(i_n);
	}

	// now for each above cell check if there is already a hit corresponding to that threshold cell in the destination list
	for(MPILib::Index i = 0; i < vec_above.size(); i++ ){
		Coordinates nearest_th = ths[vec_close[i]];

		// if there is, increase the count by the hit of the above cell
		// if there isn't insert a hit with a count of the above cell

		bool b_hit = false;
		MPILib::Index i_list = 0;
		for(MPILib::Index i = 0; i < list_ret._destination_list.size(); i++){
			const Hit& h = list_ret._destination_list[i];
			if (h._cell[0] == nearest_th[0] && h._cell[1] == nearest_th[1]){
				b_hit = true;
				i_list = i;
			}
		}
		if (b_hit){
			list_ret._destination_list[i_list]._prop += vec_above[i]._prop;
		} else {
			// if there isn't insert a hit with a count of the above cell
			Hit new_hit;
			new_hit._cell = nearest_th;
			new_hit._prop = vec_above[i]._prop;
			list_ret._destination_list.push_back(new_hit);
		}

	}

	return list_ret;
}
