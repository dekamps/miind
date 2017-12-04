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
#include <fstream>
#include <algorithm>
#include "ConstructResetMapping.hpp"

void TwoDLib::ConstructResetMapping
(
	std::ostream& ost,
	const TwoDLib::Mesh& mesh,
	const vector<TwoDLib::Coordinates>& ths,
	const vector<TwoDLib::Coordinates>& thres,
	double tr_w,
	TransitionMatrixGenerator* pgen
){

	std::cout << "zopa: " << tr_w << std::endl;
	ost << "<Mapping type=\"Reset\">\n";
	// can't assume sorting
	vector<Coordinates> ressort = thres;
	std::sort(ressort.begin(),ressort.end(),[&mesh](const Coordinates& c1, const Coordinates& c2)
			{
				return mesh.Quad(c1[0],c1[1]).Centroid()[1] < mesh.Quad(c2[0],c2[1]).Centroid()[1];
			}
	);

	// loop over all threshold bins
	for ( const Coordinates& c: ths){

		// determine the translated reset point
		double w_trans = mesh.Quad(c[0],c[1]).Centroid()[1] + tr_w;

		// find the first element that has a w value larger than the translated point
		auto it = std::find_if(ressort.begin(),ressort.end(),[w_trans,&mesh](const Coordinates& c)
				{
					return w_trans < mesh.Quad(c[0],c[1]).Centroid()[1];
				});

		// if that is not found mass should be added to the largest element
		if (it == ressort.end()){
		  ost << c[0] << "," << c[1] << "\t" << ressort.back()[0] << "," << ressort.back()[1] << "\t1.0\n";
		  
		} else
		  if (it == ressort.begin())
		    ost << c[0] << "," << c[1] << "\t" << ressort[0][0] << "," << ressort[0][1] << "\t1.0\n";
		  else {
			  // Here we are guaranteed that it and it-1 point to something.
		    auto itmin = it-1;
		    double w_min =  mesh.Quad(itmin->operator[](0),itmin->operator[](1)).Centroid()[1];
		    double w_max =  mesh.Quad(it->operator[](0),it->operator[](1)).Centroid()[1];
		    double f1 = (w_trans - w_min)/(w_max - w_min);
		    double f2 = (w_max - w_trans)/(w_max - w_min);
		    ost << c[0] << "," << c[1] << "\t" << itmin->operator[](0) << "," << itmin->operator[](1) << "\t" << f1 << "\n";
		    ost << c[0] << "," << c[1] << "\t" << it->operator[](0)    << "," << it->operator[](1)    << "\t" << f2 << "\n";
		  }
	}
	ost << "</Mapping>\n";
}
