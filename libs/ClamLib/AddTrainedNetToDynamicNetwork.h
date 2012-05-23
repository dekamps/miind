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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_CLAMLIB_ADTTRAINEDNETTODYNAMICNETWORK_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_ADTTRAINEDNETTODYNAMICNETWORK_INCLUDE_GUARD

#include "../DynamicLib/DynamicLib.h"
#include "AbstractCircuitCreator.h"
#include "BasicDefinitions.h"
#include "TrainedNet.h"
#include "WeightList.h"


using DynamicLib::AlgorithmGrid;
using DynamicLib::D_AbstractAlgorithm;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::D_RateFunctor;
using DynamicLib::Efficacy;
using DynamicLib::NodeState;
using DynamicLib::NodeType;
using DynamicLib::Rate;
using DynamicLib::RateFunction;
using DynamicLib::SimulationRunParameter;
using DynamicLib::SpatialPosition;
using DynamicLib::Time;
using DynamicLib::WilsonCowanParameter;
using NetLib::ConstNodeIterator;
using SparseImplementationLib::D_AbstractSparseNode;
using StructnetLib::OrientedPattern;
using StructnetLib::PhysicalPosition;

namespace ClamLib {


	class AddTNToDN
	{
	public:

		typedef ConstNodeIterator<TrainedNet::Implementation::NodeType> node_iterator;
		typedef D_AbstractSparseNode::const_predecessor_iterator const_predecessor_iterator;
		typedef auto_ptr<D_AbstractAlgorithm> algorithm_pointer;
		typedef auto_ptr<NodeType> type_pointer;

		//! Convert a SpatialConnectionist to part of a DynamicNetwork
		void Convert
		(
			const TrainedNet&,					//!< Static net (TrainedNet) to be converted into a DynamicNet
			const Pattern<Rate>&,				//!< Input pattern, specifying the rates that will be offered to the input layer from t = 0 onwards
			const AbstractCircuitCreator&,		//!< The CircuitCreator determines how an ANN node from the static net will be expanded into a circuit
			D_DynamicNetwork*,					//!< DynamicNetwork in which the static one will be converted
			RateFunction = 0,					//!< If this arguments is present, all non-zero elements of the input pattern will be generated according to this function
			const SpatialPosition& = NO_OFFSET  //!< The SpatialPosition of every node in the DynamicNetwork will be offset by this value
		);

		SpatialPosition 
			ConvertPhysicalPositionToSpatialPosition
			(
				const PhysicalPosition&
			) const;

		const vector<CircuitInfo>&
			CircuitInfoVector() const;

	private:

		class InvertableFunctor : public D_AbstractAlgorithm
		{
		public:

			InvertableFunctor
			(
				RateFunction,
				const WilsonCowanParameter&
			);

			virtual ~InvertableFunctor();

			virtual D_AbstractAlgorithm* Clone() const;

			virtual Time CurrentTime() const;

			virtual Rate CurrentRate() const;

			virtual bool Dump(ostream&) const;

			virtual bool Configure
			(
				const SimulationRunParameter&
			);

			virtual bool EvolveNodeState
				(
					predecessor_iterator,
					predecessor_iterator,
					Time
				);

			virtual string LogString() const;

			virtual AlgorithmGrid Grid() const;

			virtual NodeState  State() const;

		private:

			const WilsonCowanParameter _par;

			RateFunction _p_rate_function;
			Time		 _current_time;
			Rate		 _current_rate;
		};

/*
		AbstractCircuitCreator* DetermineSquashingMode
		(
			const AbstractSquashingFunction&,
			const SpatialPosition&
		);
*/
		bool IsInputNeuron
		(
			const PhysicalPosition&
		) const;

		bool IsOutputNeuron
		(
			const PhysicalPosition&
		) const;

		bool IsSymmetricSquashingFunction
		(
			const AbstractSquashingFunction&
		) const;

		void AssociateInputFieldWithInputNodes
		(
			const Pattern<Rate>&
		);

		void AddRateNode
		(
			NodeId,
			Rate 
		);

		void AddZeroRateNode
		(
			NodeId
		);

		void RetrieveParameter
		(
			NodeId,
			WilsonCowanParameter*,
			CircuitInfo*
		) const;

		void InsertDynamicNetWeights();
		void InsertWeight
		(
			const CircuitInfo&, 
			const CircuitInfo&, 
			Efficacy
		);

		D_DynamicNetwork*					_p_dnet;
		const TrainedNet*					_p_net;
		const D_AbstractAlgorithm*			_p_exc_alg;
		const D_AbstractAlgorithm*			_p_inh_alg;
		boost::shared_ptr<AbstractCircuitCreator>	_p_creator;

		RateFunction						_p_rate_function;
		std::vector<CircuitInfo>			_vec_weight_list;

	};

} // end of ClamLib


#endif // include guard
