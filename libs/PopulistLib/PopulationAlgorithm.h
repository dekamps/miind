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
#ifndef _CODE_LIBS_POPULISTLIB_POPOULATION_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_POPOULATION_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "BasicDefinitions.h"
#include "OrnsteinUhlenbeckConnection.h"
#include "PopulationGridControllerCode.h"
#include "PopulistParameter.h"
#include "OrnsteinUhlenbeckParameter.h"
#include <sstream>

using DynamicLib::AbstractAlgorithm;
using DynamicLib::AlgorithmGrid;
using DynamicLib::DynamicNetwork;
using DynamicLib::DynamicNetworkImplementation;
using DynamicLib::DynamicNode;
using DynamicLib::NodeState;
using DynamicLib::RateAlgorithm;
using DynamicLib::RateFunctor;
using DynamicLib::ReportValue;
using DynamicLib::SimulationRunParameter;
using SparseImplementationLib::SparseNode;

namespace PopulistLib {

	class PopulistSpecificParameter;
	//! \page population_algorithm Using population density techniques for modelling populations of LIF neurons.
	//! \section population_introduction Introduction

	//! PopulationAlgorithm implements the simulation algorithm for leaky-integrate-and-fire neurons
	//! as developed by de Kamps (2003,2006,2008).
	//!
	//! Examples of the usage of population density networks is described in 'The state of MIIND'
	//! The rest of the documentation is concerned with C++ aspects.
	//! PopulationAlgorithm implements the interaction of neuronal dynamics and a master equation (M-equation).
	//! By default the neuronal dynamics is leaky-integrate-and-fire dynamics and the M-equation that of a reentrant
	//! Poisson process (de Kamps, 2006). The PopulationAlgorithm is responsible for interfacing with the DynamicNode
	//! and maintains the state, i.e. represents the population density. The PopulationGridControl is responsible
	//! for the implementation of the Poisson process, by means of a ZeroLeakEquation object, and also for keeping
	//! track of the rescaling of the desnity profile due to leaky-integrate-and-fire dynamics. 
	//!
	//! Later, it was established that the 1D Markov process, described in Muller et al. (2007)
	//! http://dx.doi.org/10.1162/neco.2007.19.11.2958
	//! can also be handled if the M-equation is defined differently. Therefore a ZeroLeak object was introduced for the
	//! solution of the M-equation, which now is a template parameter of the PopulationGridController. The LIFZeroLeakEquations
	//! object solves the zero leak equations defined in Sirovich (2003) and de Kamps (2006). The OneDMZeroLeakEquations object
	//! solves the 1DM process defined by Muller et al. (2007). The OneDMAlgorithm is now deprecated.

	//! \section population_class_interplay The Interplay between PopulationAlgorithm, PopulationgridController, AbstractZeroLeakEquations and Convertor objects (LIF neurons).
	//! This section applies to leaky-integrate-and-fire neurons.
	//!
	//! \subsection high High level overview
	//! The essence of the population density methods implemented in the PopulistLibrary is that probability density is represented in a coordinate
	//! frame that co-moves with the neuronal dynamics. The mathematical detals are set out in the references above. In terms of implementation,
	//! this means that a neuronal population that receives no input is represented by a constant density profile. The density
	//! does not change but the values of the neuronal state variables associated with a particular density bin changes, so that
	//! when the density profile is interpreted using the the changing state variable associated with each bin, the constant density
	//! profile actually represents a density profile that changes according to neuronal dynamics. The density in each, however,
	//! does not change. Changes in the density representation itself can only come from external input spikes which causes density
	//! to move from one bin to  another. In technical terms, it is input spikes  that causes jumps from one characteristic to another.
	//! This zero leak jump process (the terminology ZeroLeak comes from leaky-integrate-and-fire behaviour, but will be maintained to 
	//! indicate processes that need to be considered after neuronal dynamics has been 'transformed away').  
	//!
	//! For leaky-integrate-and-fire (LIF) neurons, the process of maintaining a moving interpretation for a constant density profile
	//! is performed by the PopulationGridController. It maintains a reference to the AlgorithmGrid that represents the density profile 
	//! itself. It adapts the interpretation of the density profile, but only when needed, for example when a Report is due.
	//! The PopulationGridController also calls the AbstractZeroLeakEquation which implements the jump process.
	//! The jump process must be carried out on the the AlgorithmGrid, and the parameters for the jump process are both dependent on
	//! input parameters (input frequency, synaptic effacies, etc), but also the position in the density profile, since input parameters
	//! need to be transformed to the comving density profile as well. For this reason, the PopulationGridController internally
	//! calls AbstractZeroLeakEquations::AdaptParameters with predecessor_iterator arguments which give full information on all
	//! the populations connecting onto this one. In the standard PopulationAlgorithm the PopulationGridController maintains
	//! a current scale factor (_delta_v), which is communicated to the AbstractZeroLeakEquations. This is sufficient for
	//! AbstractZeroLeakEquations to perform local parameter adaptations. In CharacteristicAlgorithm this will be more complex.
	//! 
	//! \subsection aze AbstractZeroLeakEquations
	//! The AbstractZeroLeakequations delegates input interpretation to a LIFConvertor, which interprets the external
	//! input from other populatons and converts this into input rates and input steps. The size of these input steps is maintained
	//! in an InputSetParameter, which is communicated to the concrete AbstractCirculantSolver and the concrete AbstractNonCirculantSolver.
	//! There are two basic variants for running the circulant and non circulant solver, INTEGER and FLOATING_POINT.
	//! The developer of a concrete AbstractZeroLeakEquations type is responsible for chosing between those two modes. The distinction
	//! of the two originates from the fact that the circulant and non circulant solvers describe transport between bins. It is easier
	//! to develop these algorithms on the assumption that transport is from exactly one bin to exactly one other bin, and these algorithms
	//! run faster. In the normal running of for example a single static synaptic effacy this leads to errors, as described in (de Kamps, 2006)
	//! So, later FLOATING_POINT versions of the algorithm were developed, which are more diifculut to implement and slightly less
	//! efficient. In some cases, such as the diffusion approximation one is free to chose the jump size, and one can run  the INTEGER
	//! version of the algorithm. In the current version of the algorithm, the full information on all input populations is passed
	//! into the PopulationAlgorithm, and the passed on to a convertor object, which fills InputParameterSet. AbstractZeroLeakEquations
	//! then use these to control the AbstractCirculantSolver and the AbstractNonCirculantSolver object to produce a solution.
	//!
	//! Neither the InputParameterSet, nor the convertor object should be members of the PopulationAlgorithm, because this would tie
	//! that algorithm to a certain interpretation of the input. In the OneDMAlgorithm, for example, PopulationGridController is used
	//! successfully, even though the interpretation of the input bears no relation to the circulant solver structure used for LIF neurons.
	//! How input is interpreted in terms of input is documented in LIFConvertor. How the solver algorithms then use these parameters,
	//! which are communicated by a shared reference to a InputSetParameter instance
	//! is documented in AbstractCirculantSolver and AbstractNonCirculantSolver.
	
	//! As there are different variation for interpreting the input (adding to a Gaussian white noise, as indvidual contributions), the LIFConvertor can, depending
	//! on demand produce a single input rate and bin distance, or create lists of those.
	//! The jump algorithm is then carried out, once or multiple times based on these list. For each list the jump algorithm
	//! must be solved separately. There are different ways of doing this, and this is where AbstractZeroLeakEquations specialise.
	//! T

	//! A single jump algorithm step for density corresponding to neurons that do not cross threshold is carried out by an AbstractNonCirculantSolver,
	//! whereas density corresponding to neurons that have crossed threshold are calculated by an AbstractCirculantSolver.
	//! This structure is remaining in place even if the  concrete CirculantSolver may have little in common with the original
	//! CirculantSolverAlgorithm. For example, reintroducing density at the reset bin after refraction is also handled
	//! by the AbstractCirculantSolver.

	//! The

	template <class Weight>
	class PopulationAlgorithm_ : public AbstractAlgorithm<Weight> {
	public:

		//!
		typedef typename AbstractAlgorithm<Weight>::predecessor_iterator predecessor_iterator;
		//! An algorithm should export its parameter type
		typedef PopulistParameter Parameter;

		//! Construct an Algorithm from a stream
		PopulationAlgorithm_
		(
			istream&
		);

		//! Create a PopulistAlgorithm with settings defined in a PopulistParameter
		PopulationAlgorithm_
		(
			const PopulistParameter&
		);

		//! copy constructor
		PopulationAlgorithm_
		(
			const PopulationAlgorithm_<Weight>&
		);

		//! virtual destructor
		virtual ~PopulationAlgorithm_();

		//! 
		bool Configure(const SimulationRunParameter&);

		//! Evolve the node's state
		virtual bool EvolveNodeState
		(
			predecessor_iterator, 
			predecessor_iterator, 
			Time
		);

		//! This algorithm is dependent on synchronous updating, therefore the following function is overloaded
		virtual bool CollectExternalInput
		(
			predecessor_iterator,
			predecessor_iterator
		);

		//! Current time as maintained by the algorithm
		virtual Time CurrentTime() const;

		//! Give the current output rate
		virtual Rate CurrentRate() const;

		//! Provide a copy of the momentary grid
		virtual AlgorithmGrid Grid() const;

		//! State of the current algorithm exported as a NodeState
		virtual NodeState State() const;

		//! 
		virtual string LogString() const;

		//! Provide a clone of this algorithm
		virtual PopulationAlgorithm_<Weight>* Clone() const {return new PopulationAlgorithm_<Weight>(*this);}

		//! Dump the current state of the algorithm
		bool Dump(ostream&) const;

		//! Give the potential that corresponds to a bin number at a specific moment
		Potential BinToCurrentPotential(Index) const;

		//! Give the bin number that momentarily corresponds to a potential
		Index CurrentPotentialToBin(Potential) const;

		virtual bool Values() const {return true;}

		//! Standard tag for use in XML files
		virtual string Tag() const;

		//! Writing towards stream
		virtual bool ToStream(ostream&) const;

		//! Reading from stream
		virtual bool FromStream(istream&);

		VALUE_RETURN

	private:

		void						StripHeader			(istream&);
		void						StripFooter			(istream&);

		PopulationParameter			ParPopFromStream	(istream&);
		PopulistSpecificParameter	ParSpecFromStream	(istream&);

		void			WriteConfigurationToLog	();
		void			Embed					();

		VALUE_MEMBER

		PopulationParameter								_parameter_population;
		PopulistSpecificParameter						_parameter_specific;
		mutable ostringstream							_stream_log; // before AlgorithmGrid, which receives a pointer to this stream
		AlgorithmGrid									_grid;
		PopulationGridController<Weight>				_controller_grid;
		Time											_current_time;
		Rate											_current_rate;


	}; // end of PopulationAlgorithm

	template  <class Weight>
	ostream& operator<<(ostream& s, const PopulationAlgorithm_<Weight>& alg){alg.ToStream(s); return s;}

	// default algorithm is with PopulistConnection
	typedef PopulationAlgorithm_<OrnsteinUhlenbeckConnection>						PopulationAlgorithm;

	typedef RateAlgorithm<PopulationConnection>										Pop_RateAlgorithm;
	typedef RateFunctor<PopulationConnection>										Pop_RateFunctor;
	typedef SparseNode<double, PopulationConnection>								Pop_DynamicNode;
	typedef DynamicNetwork<DynamicNetworkImplementation <PopulationConnection> >	Pop_Network;
}

#endif // include guard

