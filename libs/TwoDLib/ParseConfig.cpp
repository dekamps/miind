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
#include <sstream>
#include <string>

#include "ParseConfig.hpp"
#include "FiducialElement.hpp"

TwoDLib::Config TwoDLib::ParseConfig(std::ifstream& ifst){

	std::string dummy;
	unsigned int n_points;
	double tr_v, tr_w;
	vector<double> vec_v(4);
	vector<double> vec_w(4);

	// V_reset
	double V_reset, w_reset;
	ifst >> dummy;
	if (dummy == "V_reset:"){
		ifst >> V_reset >> w_reset;
	}
	else
		throw TwoDLib::TwoDLibException("Config file can't be parsed.");

	// V_threshold
	double V_threshold;
	ifst >> dummy;
	if (dummy == "V_threshold:"){
		ifst >> V_threshold;
	}
	else
		throw TwoDLib::TwoDLibException("Config file can't be parsed.");

	ifst >> dummy;
	std::vector<TwoDLib::Quadrilateral> stationary_bins = ParseStationary(ifst);


	std::vector<TwoDLib::ProtoFiducial> vec_fiducial;
	while(ifst){
		getline(ifst,dummy);

		// it is legal not to have any fiducial elements
		if (dummy.length() == 0)
			break;
		TwoDLib::Overflow overflow;
		if (dummy.find("Leak") != std::string::npos)
			overflow = TwoDLib::LEAK;
		else if (dummy.find("Contain") != std::string::npos)
			overflow = TwoDLib::CONTAIN;
		else
			throw TwoDLib::TwoDLibException("Config file can't be parsed");

		ifst >> vec_v[0] >> vec_v[1] >> vec_v[2] >> vec_v[3];
		ifst >> vec_w[0] >> vec_w[1] >> vec_w[2] >> vec_w[3];

		ifst >> dummy; // absorb \n
		TwoDLib::Quadrilateral quad(vec_v,vec_w);
		TwoDLib::ProtoFiducial fid(quad,overflow);
		vec_fiducial.push_back(fid);
	}

	TwoDLib::Config config(n_points,tr_v,tr_w,V_reset,w_reset,V_threshold,stationary_bins,vec_fiducial);
	return config;
}

std::vector<TwoDLib::Quadrilateral> TwoDLib::ParseStationary(std::ifstream& ifst){
	std::vector<TwoDLib::Quadrilateral> vec_quad;
	string dummy;
	getline(ifst,dummy);
	if (dummy.find("bin") == std::string::npos)
		throw TwoDLib::TwoDLibException("Expected reversal bin tag");

	vector<double> vec_v(4);
	vector<double> vec_w(4);

	while(ifst){
		ifst >> vec_v[0] >> vec_v[1] >> vec_v[2] >> vec_v[3];
		if (!ifst) break;
		ifst >> vec_w[0] >> vec_w[1] >> vec_w[2] >> vec_w[3];
		TwoDLib::Quadrilateral quad(vec_v,vec_w);
		vec_quad.push_back(quad);
	}
	ifst.clear();

	return vec_quad;
}

