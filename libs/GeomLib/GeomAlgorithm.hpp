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

#ifndef _CODE_LIBS_GEOMLIB_GEOMALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_GEOMALGORITHM_INCLUDE_GUARD

#include <boost/circular_buffer.hpp>
#include <MPILib/include/algorithm/AlgorithmInterface.hpp>
#include "GeomParameter.hpp"
#include "NumericalMasterEquationCode.hpp"

using MPILib::algorithm::AlgorithmGrid;
using MPILib::algorithm::AlgorithmInterface;
using MPILib::DelayedConnection;
using MPILib::Rate;
using MPILib::SimulationRunParameter;
using MPILib::Time;

namespace GeomLib {
  //! Population density algorithm based on Geometric binning: http://arxiv.org/abs/1309.1654

  //! Population density techniques are used to model neural populations. See
  //! http://link.springer.com/article/10.1023/A:1008964915724 for an introduction.
  //! This algorithm uses a geometric binning scheme as described in http://arxiv.org/abs/1309.1654 .
  //! The rest of this comment will describe the interaction with the MIIND framework and the
  //! objects that it requires.
  //! \section label_geom_intro Introduction
  //! GeomAlgorithm inherits from AlgorithmInterface. This implies that a node in an MPINetwork
  //! can be configured with this algorithm, and that network simulations can be run, where
  //! each node represents a neuronal population. The simulations are equivalent to spiking neuron
  //! simulations of point model neurons, such as performed by NEST, with some caveats. GeomAlgorithm
  //! deals with one dimensional point models, such as leaky-integrate-and-fire, quadratic-integrate-and-fire
  //! and exponential-integrate-and-fire. 2D models such adaptive-exponential-integrate-and-fire and conductance
  //! based models will be handled by another algorithm, which is in alpha stage. GeomAlgorithm
  //! is instantiated using a GeomParameter, which specifies, among other things, the neuronal model
  //! and its parameter values, in the form of an AbstractOdeSystem. From the user perspective, this is
  //! the most important, and the documentation of AbstractOdeSystem, and more the important the
  //! <a href="http://miind.sf.net/tutorial.pdf">MIIND tutorial</a> is the first port of call. In the remainder
  //! of this documentation section the interaction of GeomAlgorithm with other objects will be discussed.
  //!
  //! \section label_geom_initialization The Initialization Sequence
  //!
  //! The creation sequence is lengthy although the user only sees the first two stages, represented in blue.
  //! NeuronParameter defines quantities common to most neuronal models, such as membrane potential,
  //! membrane time constant, threshold, etc. OdeParameter accepts a NeuronParameter, and other parameters
  //! that set the dimension of the grid. LifNeuralDynamics defines the neuronal model itself in terms of
  //! a method 	LifNeuralDynamics::EvolvePotential(MPILib::Potential,MPILib::Time), which describes
  //! how a potential evolves over time under the dynamics of the neuronal model, in this case leaky-integrate-and
  //! fire dynamics. For most users the dynamics of a model will already have been defined. The introduction
  //! of novel dynamics requires an overload of the corresponding function of AbstractNeuralDynamics.
  //! LifOdeSystem is a representation of the grid itself. Note that for spiking neural dynamics another
  //! grid representation is needed: SpikingNeuralDynamics. This difference will be removed in the future.
  //! The constructor call requires the following initialization sequence.
  //! \msc["Main loop"]
  //!     NeuronParameter, OdeParameter, LifNeuralDynamics, LifOdeSystem, GeomParameter;
  //!
  //!   NeuronParameter=>OdeParameter [label="argument", linecolor="blue"];
  //!   OdeParameter=>LifNeuralDynamics[label="argument", linecolor="blue"];
  //!   LifNeuralDynamics=>LifOdeSystem[label="argument"];
  //!   LifOdeSystem=>GeomParameter[label="argument"];
  //! \endmsc
  //!
  //! section label_geom_creation The Creation Sequence
    template <class WeightValue>
	class GeomAlgorithm : public AlgorithmInterface<WeightValue>  {
	public:

		typedef GeomParameter Parameter;
		using AlgorithmInterface<WeightValue>::evolveNodeState;

		//! Standard way for user to create algorithm
		GeomAlgorithm
		(
			const GeomParameter&
		);

		//! Copy constructor
		GeomAlgorithm(const GeomAlgorithm&);

		//! virtual destructor
		virtual ~GeomAlgorithm();

		/**
		 * Cloning operation, to provide each DynamicNode with its own
		 * Algorithm instance. Clients use the naked pointer at their own risk.
		 */
		virtual GeomAlgorithm* clone() const;

		/**
		 * Configure the Algorithm
		 * @param simParam The simulation parameter
		 */
		virtual void configure(const SimulationRunParameter& simParam);


		virtual void evolveNodeState(const std::vector<Rate>& nodeVector,
				const std::vector<WeightValue>& weightVector, Time time);

		virtual void prepareEvolve(const std::vector<Rate>& nodeVector,
				const std::vector<WeightValue>& weightVector,
				const std::vector<MPILib::NodeType>& typeVector);


		/**
		 * The current time point
		 * @return The current time point
		 */
		virtual MPILib::Time getCurrentTime() const;


		/**
		 * The calculated rate of the node
		 * @return The rate of the node
		 */
		virtual Rate getCurrentRate() const;

		/**
		 * Stores the algorithm state in a Algorithm Grid
		 * @return The state of the algorithm
		 */
		virtual AlgorithmGrid getGrid(NodeId) const;

	private:

		bool  IsReportDue() const;

    const GeomParameter	      			_par_geom;
    AlgorithmGrid	      				_grid;
    unique_ptr<AbstractOdeSystem>      	_p_system;
    unique_ptr<AbstractMasterEquation>	_p_zl;

    bool    _b_zl;
    Time    _t_cur;
    Time    _t_step;
    Time    _t_report;

    mutable Number _n_report;

  };
}

#endif // include guard

