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
#ifndef _CODE_LIBS_TWODLIB_PARSECONFIG_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_PARSECONFIG_INCLUDE_GUARD

#include <vector>
#include <fstream>
#include "FiducialElement.hpp"
#include "TwoDLibException.hpp"

namespace TwoDLib {


	struct Config{
		unsigned int _n_points;
		double _tr_v;
		double _tr_w;
		double _V_reset;
		double _w_reset;
		double _V_threshold;

		std::vector<Quadrilateral> _stationary_bins;
		std::vector<ProtoFiducial> _vec_fiducial;

		Config
		(
			unsigned int n_points,
			double tr_v,
			double tr_w,
			double V_reset,
			double w_reset,
			double V_threshold,
			const std::vector<Quadrilateral>& stationary_bins,
			const std::vector<ProtoFiducial>& vec_fiducial
		):
		_n_points(n_points),
		_tr_v(tr_v),
		_tr_w(tr_w),
		_V_reset(V_reset),
		_w_reset(w_reset),
		_V_threshold(V_threshold),
		_stationary_bins(stationary_bins),
		_vec_fiducial(vec_fiducial)
		{
		}

	};

	TwoDLib::Config                     ParseConfig     (std::ifstream&);
	std::vector<TwoDLib::Quadrilateral> ParseStationary (std::ifstream&);

}
#endif // include guard
