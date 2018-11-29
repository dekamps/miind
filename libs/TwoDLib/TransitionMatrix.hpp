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
#ifndef _CODE_LIBS_TWODLIB_TRANSITIONMATRIX_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_TRANSITIONMATRIX_INCLUDE_GUARD

#include <string>
#include <vector>

#include "Coordinates.hpp"
#include "Redistribution.hpp"
namespace TwoDLib {

	class TransitionMatrix {
	public:

		struct Redistribution {
			Coordinates _to;
			double 	    _fraction;
		};

		struct TransferLine {
			Coordinates                 _from;
			std::vector<Redistribution> _vec_to_line;
		};

		TransitionMatrix();

		//! Requires a ".mat" file
		TransitionMatrix(const std::string&);

		TransitionMatrix(const std::vector<TwoDLib::Redistribution>&);

		bool isGeneratedFroimResetValues() const { return _generated_from_reset; }

		friend class Master;
		friend class MasterOMP;

		bool SelfTest(double) const;

		const std::vector<TransferLine>& Matrix() const { return _vec_line; }

		//! A jump could be in two dimensions, but for now we assume that a  jump is either
		//! in v, or in the second parameter, for example in a conductance-based neuron, but not both.
		double Efficacy() const { return _tr_v ? _tr_v : _tr_w; }

	private:

		std::vector<TransferLine>	_vec_line;
		double						_tr_v;
		double						_tr_w;

		bool _generated_from_reset;

	};

}

#endif
