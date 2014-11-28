// Copyright (c) 2005 - 2014 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_ABSTRACTODESYSTEM_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_ABSTRACTODESYSTEM_INCLUDE_GUARD

#include <vector>
#include "AbstractNeuralDynamics.hpp"
//#include "BasicDefinitions.hpp"
#include "OdeParameter.hpp"

using std::vector;

namespace GeomLib {

	//! Base class for Ode solvers for use in QIFAlgorithm

	//! QIFAlgorithm relies on externally provided solver classes for the system of ordinary differential equations.
	//! AbstractOdeSystem is the base class for such solvers. These solvers evolve the phase space in which probability densities
	//! are represented. The decision to pass a naked reference to the interpretation array, rather than a well defined
	//! C++ object reflects the vast difference between different algorithms in the use of containers. TO DO: consider wrapper
	//! class for this.

	class NumericalMasterEquations;
	class AbstractOdeSystem {
	public:

		AbstractOdeSystem
		(
			const AbstractNeuralDynamics& 						//! NeuralDynamics object
		);

		//! copy constructor
		AbstractOdeSystem(const AbstractOdeSystem& sys);

		//! pure virtual destructor for base class
		virtual ~AbstractOdeSystem() = 0;
		
		//! Every sub class defines its own evolution. 
		virtual void 
			Evolve
			(
				MPILib::Time
			) = 0;

		MPILib::Time TStep() const {return _t_step; }

		virtual AbstractOdeSystem* Clone() const  = 0;

		//! Access to the  OdeParameter of the system. It is often time-critical, therefore implemented as reference return.
		const OdeParameter& Par() const { return _par; }


		MPILib::Time CurrentTime() const { return _t_current; }

		virtual MPILib::Rate CurrentRate() const = 0;

		virtual Potential DCContribution() const { return 0;}

		Number NumberOfBins() const {return _number_of_bins; }

		vector<MPILib::Potential>& InterpretationBuffer()             { return _buffer_interpretation; }

		const vector<MPILib::Potential>& InterpretationBuffer() const { return _buffer_interpretation; }

		vector<MPILib::Potential>& MassBuffer()                       { return _buffer_mass; }

		const vector<MPILib::Potential>& MassBuffer() const           { return _buffer_mass; }

		Index IndexResetBin() const {return _i_reset;}

		Index FindBin(Potential) const;

		Index MapPotentialToProbabilityBin(Index i) const { assert(i < _map_cache.size()); return _map_cache[i]; }

		//! make sure that the interpretation array is up-to-date
		void PrepareReport
		(
			double*,
			double*
		) const;

	protected:

		boost::shared_ptr<AbstractNeuralDynamics> _p_dyn;
		const string		_name_namerical;
		MPILib::Time				_t_period;
		MPILib::Time 				_t_step;

		const OdeParameter& _par;
		vector<MPILib::Potential>	_buffer_interpretation;
		vector<MPILib::Density>		_buffer_mass;

		Index		       	_i_reset;
		Index				_i_reversal;
		MPILib::Time		       	_t_current;

		vector<Index>		_map_cache;

	private:

		Index InitializeResetBin    () const;
		Index InitializeReversalBin () const;

		vector<MPILib::Density> InitializeDensity	() const;

		void NormaliseDensity  				(vector<MPILib::Density>*) const;
		void InitializeGaussian				(vector<MPILib::Density>*) const;
		void InitializeSingleBin       		(vector<MPILib::Density>*) const;

		Number _number_of_bins;
	};
}

#endif // include guard

