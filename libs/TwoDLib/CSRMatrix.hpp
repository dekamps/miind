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

#ifndef _CODE_LIBS_TWODLIB_CSRMATRIX_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_CSRMATRIX_INCLUDE_GUARD

#include <MPILib/include/TypeDefinitions.hpp>
#include "Ode2DSystem.hpp"
#include "TransitionMatrix.hpp"


namespace TwoDLib {


	class CSRMatrix {
	public:

		//! It is assumed that the Mesh and Config file used to produce the Ode2DSystem
		//! are the same that were used to produce the TransitionMatrix
		CSRMatrix
		(
			const TransitionMatrix&,  //!< TransitionMatrix as read in from a .mat file
			const Ode2DSystem&        //!< Ode2DSystem that this matrix will operate on
		);


		//! out += Mv; ranges assert checked
		void MV(vector<double>& out, const vector<double>& v);

		//! Performs a matrix-vector multiplication, taking into account the current density mapping
		void MVMapped(vector<double>&, const vector<double>&, double) const;

		//! Each matrix corresponds to a well defined jump
		double Efficacy() const {return _efficacy; }

	private:

		void Initialize(const TransitionMatrix&);
		void Validate(const TransitionMatrix&);
	  void CSR(const vector<vector<MPILib::Index> >&, const vector<vector<double> >&);

		const Ode2DSystem& _sys;
		const double       _efficacy; // efficacy used in the generation of the TransitionMatrix

		std::vector<double>       _val;
		std::vector<unsigned int> _ia;
		std::vector<unsigned int> _ja;
		std::vector<Coordinates>  _coordinates;

	};
}

#endif /* CSRMATRIX_HPP_ */
