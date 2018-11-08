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

#include <iostream>
#include <algorithm>
#include <vector>
#include "CheckSystem.hpp"
#include "Ode2DSystem.hpp"


void TwoDLib::CheckSystem
(
	const TwoDLib::Ode2DSystem& sys,
	const TransitionMatrix& mat,
	const vector<TwoDLib::Redistribution>& reversal_map,
	const vector<TwoDLib::Redistribution>& reset_map,
	double V_th
)
{
	// Check whether mapping is one-on-one

	std::vector<int> vec_map(sys._vec_mass.size(),0);
	int n_strips = sys._mesh.NrStrips();

	for(int i = 0; i < n_strips; i++){
		int n_cells = sys._mesh.NrCellsInStrip(i);
		for (int j = 0; j < n_cells; j++)
			vec_map[sys.Map(i,j)] = 1;
	}

	for (int i: vec_map)
		if (vec_map[i] != 1)
			std::cout << "Mapping not on-on-one" << std::endl;

	// Check whether all probability above threshold is really removed

	vector<TwoDLib::Coordinates> vec_above = sys._mesh.findV(V_th,Mesh::ABOVE);
	for (const Coordinates& c: vec_above){
		if (sys._vec_mass[sys.Map(c[0],c[1])] != 0.)
			std::cout << "Probability above threshold" << std::endl;
	}

	vector<TwoDLib::Coordinates> vec_equal = sys._mesh.findV(V_th,Mesh::EQUAL);
	for (const Coordinates& c: vec_equal){
		if (sys._vec_mass[sys.Map(c[0],c[1])] != 0.)
			std::cout << "Probability on threshold" << std::endl;
	}


	// Check whether there are transitions above threshold, and if so whether they are
	// into the stray bins

	const vector<TwoDLib::TransitionMatrix::TransferLine>& matrix = mat.Matrix();
	for (const TwoDLib::TransitionMatrix::TransferLine& line: matrix){
		if( std::find(vec_above.begin(),vec_above.end(),line._from) != vec_above.end())
			std::cout << "Origininating from above: " << line._from[0] << " " << line._from[1] << std::endl;

		if( std::find(vec_equal.begin(),vec_equal.end(),line._from) != vec_equal.end())
			std::cout << "Origininating from equal: " << line._from[0] << " " << line._from[1] << std::endl;

		for( const TwoDLib::TransitionMatrix::Redistribution& r: line._vec_to_line){
			if ( std::find(vec_above.begin(),vec_above.end(),r._to) != vec_above.end() ){
				std::cout << "to above: " << r._to[0] << " " << r._to[1];
				std::cout << " from: " << line._from[0] << " " << line._from[1] << std::endl;
			}

			if ( std::find(vec_equal.begin(),vec_equal.end(),r._to) != vec_equal.end() ){
				std::cout << "to equal: " << r._to[0] << " " << r._to[1] << std::endl;
			    std::cout << " from: " << line._from[0] << " " << line._from[1] << std::endl;
			}
		}
	}


}
