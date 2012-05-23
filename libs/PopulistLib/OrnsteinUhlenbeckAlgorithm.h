// Copyright (c) 2005 - 2010 Marc de Kamps
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
#ifndef _CODE_LIBS_POPULISTLIB_ORNSTEINUHLENBECKALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_ORNSTEINUHLENBECKALGORITHM_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "LocalDefinitions.h"
#include "ResponseParameterBrunel.h"
#include "OrnsteinUhlenbeckConnection.h"
#include "OrnsteinUhlenbeckParameter.h"

using DynamicLib::AbstractAlgorithm;
using DynamicLib::AlgorithmGrid;
using DynamicLib::DynamicNode;
using DynamicLib::DynamicNetwork;
using DynamicLib::DynamicNetworkImplementation;
using DynamicLib::NodeState;
using DynamicLib::RateAlgorithm;
using DynamicLib::ReportValue;
using DynamicLib::SimulationRunParameter;
using SparseImplementationLib::SparseNode;

namespace PopulistLib {
	//! \page steady_state Modelling the steady state of leaky-integrate-and-fire populations
	//! \section steady-state_introduction Introduction
	//! When cortical neurons experience a barrage from other cortical neurons the input can sometimes
	//! usefully be described as a Gaussian white noise process, characterised by a mean \f$ \mu \f$ and variance
	//! \f$ \sigma \f$. The response of an entire population, where it is assumed that individual neurons
	//! each see different spike trains, but where the statistics of the input of each neuron is assumed to be the same
	//! can be evaluated for some neuronal models. For leaky-integrate-and-fire (LIF) neurons, modelling
	//! such a response can be done analytically if the input is steady (i.e. not varying over time; individual
	//! neurons will experience variation, due to their noisy input spike trains, but with steady state we indicate that
	//! \f$ \mu \f$ and \f$\sigma $\f are constant. Clearly, in the brain activation mostly is not steady state,
	//! and when a steady state description is not useful, population density techniques must be used (see \ref population_density).
	//! Often, however, investigating steady state conditions helps in understanding the structure of neural circuits.
	//! Under the assumption of steady state Gaussian white noise input the firing rate of populations can be calculated analytically.
	//! A Wilson-Cowan-like algorithm that gives the correct steady state of an Ornstein-Uhlenbeck process.
	//!
	//! The steady state of a large population of leaky-integrate-and-fire neurons can be descibed accurately
	//! by a so-called spike response function. This spike response function can be calculated analytically if
	//! the input rate on the population is high and the synaptic efficacies deliver small contributions to the
	//! membrane potential (small with respect to threshold). Under such assumptions the membrane potential distribution
	//! of the population may be approximated by a diffusion process, the Ornstein-Uhlenbeck process. The steady
	//! state response of the population can be calculated from that. Networks modelled with this algorithm will
	//! describe the steady states of networks of leaky-integrate-and-fire neurons accurately (if the populations
	//! are large), but not the transient dynamics. Use PopulistLib if this is an issue. Since the name is unwieldy,
	//! there is a typedef OUAlgorithm.
	class OrnsteinUhlenbeckAlgorithm : public AbstractAlgorithm<OrnsteinUhlenbeckConnection> {
	public:

		typedef OrnsteinUhlenbeckParameter Parameter;

		//! Creat an OUAlgorithm from a stream
		OrnsteinUhlenbeckAlgorithm
		(
			istream&
		);

		//! Create an OUAlgorithm from neuronal parameters 
		OrnsteinUhlenbeckAlgorithm
		(
			const OrnsteinUhlenbeckParameter&
		);

		//1 copy ctor
		OrnsteinUhlenbeckAlgorithm
		(
			const OrnsteinUhlenbeckAlgorithm&
		);

		//! virtual destructor
		virtual ~OrnsteinUhlenbeckAlgorithm();

		//! configure algorithm
		virtual bool Configure
		(
			const SimulationRunParameter&
		);

		//! Evolve the node state
		virtual bool EvolveNodeState
		(
			predecessor_iterator,	//!< begin() iterator of a connection for calculation of an inner product 
			predecessor_iterator,	//!< end iterator
			Time					//!< time step over which must be evolved
		);

		//! Current AlgorithmGrid
		virtual AlgorithmGrid Grid() const;

		virtual AlgorithmGrid DefaultInitialGrid() const;

		//! Current NodeState
		virtual NodeState State() const;

		virtual string LogString() const {return string("");}

		//! Cloning method
		virtual OrnsteinUhlenbeckAlgorithm* Clone() const;

		//! Current tme of the simulation
		virtual Time CurrentTime() const;
	
		//! Current output rate of the population
		virtual Rate CurrentRate() const;

		//! Can be used by a Simulation to dump the results so far (not intended for human users)
		virtual bool Dump(ostream&) const;

		//! Writing to stream
		virtual bool ToStream(ostream& ) const;

		//! Reading from stream (not implemented)
		virtual bool FromStream(istream&);

		virtual string Tag() const;

		double InnerSquaredProduct
		(
			predecessor_iterator iter_begin,
			predecessor_iterator iter_end
		) const;


	private:

		AlgorithmGrid InitialGrid() const;

		vector<double> InitialState() const;

		ResponseParameterBrunel 
			InitializeParameters
			(
				const OrnsteinUhlenbeckParameter& 
			) const;

		ResponseParameterBrunel
			InitializeParameters
			(
				istream& 
			);

		ResponseParameterBrunel					_parameter_response;
		DVIntegrator<ResponseParameterBrunel>	_integrator;
	};

	typedef DynamicNode<OrnsteinUhlenbeckConnection>	OU_DynamicNode;
	typedef OrnsteinUhlenbeckAlgorithm					OU_Algorithm;
	typedef RateAlgorithm<OrnsteinUhlenbeckConnection>	OU_RateAlgorithm;

	typedef DynamicNetwork<DynamicNetworkImplementation<OrnsteinUhlenbeckConnection> > OU_Network;

} // end of PopulistLib

#endif // include guard
