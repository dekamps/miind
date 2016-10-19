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
#include "CorrectStrays.hpp"
#include "TwoDLibException.hpp"

namespace {
	vector<TwoDLib::Hit>::iterator find_nearest(std::vector<TwoDLib::Hit>& vec_th, const TwoDLib::Hit& h){
		vector<double> dist;

		for(auto hth: vec_th){
			double d = std::pow(hth._cell[0] -  h._cell[0],2) + std::pow(hth._cell[1] - h._cell[1],2);
			dist.push_back(d);
		}

		ptrdiff_t i_min = std::distance(dist.begin(), std::min_element(dist.begin(),dist.end()));
		return vec_th.begin() + i_min;
	}
}

TwoDLib::TransitionList TwoDLib::CorrectStrays
(
	const TwoDLib::TransitionList& l,               //! list of all hits for a transition
	const vector<TwoDLib::Coordinates>& ths,     	//! list of cells on threshold
	const vector<TwoDLib::Coordinates>& above	    //! list of cells above threshold
){
	TransitionList list_ret;
	list_ret._origin = l._origin;
	list_ret._number = l._number;

	std::vector<TwoDLib::Hit> vec_th;
	std::vector<TwoDLib::Hit> vec_above;

	for(const TwoDLib::Hit& h: l._destination_list){
		if ( std::find(ths.begin(), ths.end(), h._cell) != ths.end() ){
			vec_th.push_back(h);
		}
		else
			if ( std::find(above.begin(),above.end(),h._cell) != above.end() ){
				vec_above.push_back(h);
			}
 			else
				list_ret._destination_list.push_back(h);
	}

//	if (vec_th.size() == 0 && vec_above.size() != 0){
//		std::ostringstream ost;
//		ost << "Cell: " << l._origin[0] << "," <<  l._origin[1] << ". Stray, but no threshold cell in this transition\n";
//		throw TwoDLib::TwoDLibException(ost.str());
//	}

	if (vec_th.size() > 0 ){
		for (auto a: vec_above){
			auto it = find_nearest(vec_th, a);
			it->_count += a._count;
		}
	}

	for (auto h: vec_th)
		list_ret._destination_list.push_back(h);

	return list_ret;
}
