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
#include "Ode2DSystemGroup.hpp"
#include "TransitionMatrix.hpp"


namespace TwoDLib {

	//! Calculates derivative as a matrix vector multiplication, whilst taking into acoount the
	//! current mapping of the Ode2DSystemGroup.

	class CSRMatrix {
	public:

		//! It is assumed that the Mesh and Config file used to produce the Ode2DSystemGroup
		//! are the same that were used to produce the TransitionMatrix
		CSRMatrix
		(
			const TransitionMatrix&  tmat, 		 //!< TransitionMatrix as read in from a .mat file
			const Ode2DSystemGroup&  system,     //!< Ode2DSystemGroup that this matrix will operate on
			MPILib::Index            mesh_index  //!< Index of the Mesh in the Ode2DSystemGroup that this CSRMatrix
		);


		//! out += Mv; ranges assert checked, this does not take into account the mapping by the Ode2DSystemGroup
		void MV
			(
				vector<double>& 		out_result, //!< The result of M v_in
				const vector<double>&   v_in        //!< v_in: vector to be multiplied by the TransitionMatrix m
			);

		//! Performs a matrix-vector multiplication, taking into account the current density mapping
		void MVMapped
			(
				vector<double>&        dydt, 		//!< Reference to the derivative array
				const vector<double>&  density,     //!< Reference to density array
				double				   rate         //!< Firing rate that should be applied to the derivative
			) const;

		//! Each matrix corresponds to a well defined jump
		double Efficacy() const {return _efficacy; }

		//! Expose underlying arrays Val
		const std::vector<double>& Val() const {return _val;}
		//! Expose underlying arrays Ia
		const std::vector<unsigned int>& Ia() const {return _ia;}
		//! Expose underlying arrays Ja
		const std::vector<unsigned int>& Ja() const {return _ja;}

		//! Which mesh is this matrix relating to?
		MPILib::Index MeshIndex() const {return _mesh_index; }

		//! Which offset of the mass array does this correspond to
		MPILib::Index Offset() const { return _i_offset; }

		//! Number of rows corresponding this this matrix

		MPILib::Index NrRows() const { return _ia.size() - 1; }

	private:

		void Initialize(const TransitionMatrix&, MPILib::Index);
		void Validate(const TransitionMatrix&);

		void CSR(const vector<vector<MPILib::Index> >&, const vector<vector<double> >&);

		const Ode2DSystemGroup& _sys;
		const double       _efficacy; // efficacy used in the generation of the TransitionMatrix

		std::vector<double>       _val;
		std::vector<unsigned int> _ia;
		std::vector<unsigned int> _ja;

		MPILib::Index _mesh_index;  // index of the Mesh on the Ode2DSystemGroup that this CSRMatrix is responsible for
		MPILib::Index _i_offset;    // offset of the part of the mass array that this CSRMatrix is responsible for

	};
}

#endif /* CSRMATRIX_HPP_ */
