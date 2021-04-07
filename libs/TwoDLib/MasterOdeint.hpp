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
#ifndef _CODE_LIBS_TWODLIB_MASTERODEINT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_MASTERODEINT_INCLUDE_GUARD

#include <string>
#include <boost/numeric/odeint.hpp>
#include "CSRMatrix.hpp"
#include "TransitionMatrix.hpp"
#include "Ode2DSystemGroup.hpp"
#include "MasterParameter.hpp"

typedef boost::numeric::odeint::runge_kutta_cash_karp54< std::vector<double> > error_stepper_type;

namespace TwoDLib {

	// Solver for the Poisson Master equation, using BOOST::Odeint as backend
	class MasterOdeint {
	public:

		MasterOdeint
		(
			Ode2DSystemGroup&  sys,                                     //!< The systemgroup keeps track of the density and the mapping that is updated using evolution
			const std::vector< vector<TransitionMatrix> >& vec_vec_mat, //!< TransitionMatrix matrix. First index labels the Mesh that covered by a vector of TransitionMatrix elements, second index labels the different TransitionMatrix objects operating on this mesh, which differ in the efficacy they represent
			const MasterParameter& par                                  //!< Parmeter for configuring the numerical solver
		);

		MasterOdeint(const MasterOdeint&);

		//! Communicate the firing rates to the sparse matrix implementation
		void Apply
			(
				double                                   time_step,      //!< mesh time step
				const std::vector<std::vector<double> >& rate_matrix,	 //!< rate matrix
				const vector<MPILib::Index>&	         weight_order    //!< reference to the mapping from NodeId to weight order
			);

		//! Communicate the firing rates to the sparse matrix implementation for individual neurons
		void ApplyFinitePoisson
		(
			double                                   time_step,      //!< mesh time step
			const std::vector<std::vector<double> >& rate_matrix,	 //!< rate matrix
			const vector<MPILib::Index>& weight_order    //!< reference to the mapping from NodeId to weight order
		);

		//! Number of meshes
		MPILib::Number  NrMeshes() const { return _vec_vec_csr.size(); }

		//! Number of weight matrices associated with Mesh i
		MPILib::Number  NrMatrices(MPILib::Index i) const { return _vec_vec_csr[i].size(); }

		//! Efficacy associated with matrix matrix_index from Mesh mesh_index
		double Efficacy
		(
				MPILib::Index mesh_index,	//!< Index labeling mesh
				MPILib::Index matrix_index
		) const {return _vec_vec_csr[mesh_index][matrix_index].Efficacy(); }

		//! Solution step
		void operator()
				(
					const vector<double>&,
					vector<double>&,
					const double t = 0
				);

	private:

		MasterOdeint& operator=(const MasterOdeint&);

		std::vector<std::vector<CSRMatrix> >
			InitializeCSR
			(
				const std::vector< std::vector<TransitionMatrix> >&,
				const Ode2DSystemGroup&
			);

		Ode2DSystemGroup& _sys;

		const std::vector< std::vector<TransitionMatrix> >& _vec_vec_mat; // After initialization, the original object is allowed to go out of scope; it will not be referred anymore
		const std::vector< std::vector<CSRMatrix> >         _vec_vec_csr;

		const MasterParameter	_par;
		vector<double>			_dydt;
		double					_rate;

		const std::vector<MPILib::Index>*              _p_vec_map;     // place holder for current mapping, needed in operator()
		const std::vector<std::vector<MPILib::Rate> >* _p_vec_rates;   // place holder for rate vector

	};
}

#endif // include guard
