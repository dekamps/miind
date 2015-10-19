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
#ifndef _CODE_LIBS_GEOMLIB_MATRIXMASTEREQUATION_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_MATRIXMASTEREQUATION_INCLUDE_GUARD

#include "../NumtoolsLib/NumtoolsLib.h"
#include "MasterParameter.hpp"

using MPILib::Density;
using NumtoolsLib::ExStateDVIntegrator;
using NumtoolsLib::Precision;

namespace GeomLib {

	//! derivative function for MatrixMasterEquation
	int DerivMatrixVersion( double t, const double y[], double dydt[], void* params);

	struct MatrixMasterEquation {

		//! Second independent solver for the master equation, based on a matrix representation
		//! so that conservation of probability can be checked easily

		MatrixMasterEquation
		(
			SpikingOdeSystem&,		    //! system
			Number,				    //! maximum number of iterations of the algorithm
			Index,				    //! index reset bin
			const Precision&,		    //! precision parameter for integrator
			Time,				    //! initial time
			Time				    //! time step

		);

		void Initialize
		(
			Index,					//! probability index
			const vector<InputParameterSet>&	//! input parameter vector
		);

		const SpikingOdeSystem&					_system;
		const vector<Potential>&				_vec_potential;
		vector<Density>&					_vec_density;
		Index							_i_reset;
		vector<Density>						_vec_matrix_state;

		NumtoolsLib::ExStateDVIntegrator<MasterParameter> 	_matrix_integrator;
		NumtoolsLib::QaDirty<double>				_mat_transit;
	};
}

#endif // include guard
