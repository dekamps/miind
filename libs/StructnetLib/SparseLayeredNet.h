// Copyright (c) 2005 - 2009 Marc de Kamps
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
#ifndef _CODE_LIBS_STRUCNET_BIONET_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_BIONET_INCLUDE_GUARD 


#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <string>
#include <memory>
#include "../ConnectionismLib/ConnectionismLib.h"
#include "LinkRelation.h"
#include "LayerDescription.h"
#include "NodeIdPosition.h"
#include "DenseOverlapLinkRelation.h"

using NetLib::LayeredArchitecture;
using NetLib::NetworkParsingException;
using NetLib::AbstractSquashingFunction;
using NetLib::NodeIterator;
using NetLib::Sigmoid;
using NetLib::SigmoidParameter;
using ConnectionismLib::BackpropTrainingVector;
using ConnectionismLib::Hebbian;
using ConnectionismLib::LayeredNetwork;
using std::istream;
using std::ostream;
using SparseImplementationLib::D_LayeredReversibleSparseImplementation;


namespace StructnetLib
{
	//! SpatialLayeredNet
	//! A SpatialLayeredNet (formerly BioNet) is a layered network where each Node is associated with a position

	template <class Implementation>
	class SpatialLayeredNet : public LayeredNetwork<Implementation> 
	{
	public:

		//! Recreate a spatial network that was saved on disk
		SpatialLayeredNet
		(
			istream&
		);

		//! Create a spatialnetwork from a physical description
		SpatialLayeredNet
		(
			const AbstractLinkRelation*
		);


		SpatialLayeredNet
		(
			const AbstractLinkRelation*,
			const SigmoidParameter&
		);
		SpatialLayeredNet
		(
			const AbstractLinkRelation*, 
			const AbstractSquashingFunction&
		);

		//! copy constructor
		SpatialLayeredNet
		(
			const SpatialLayeredNet<Implementation>&
		);

		//! virtual destructor
		virtual ~SpatialLayeredNet();

		//! Write a SpatialLayeredNet to a stream
		virtual bool ToStream(ostream&) const;

		const vector<LayerDescription>&  Dimensions() const;

		const PhysicalPosition&	Position(NodeId) const;
			  NodeId			Id		( const PhysicalPosition& ) const;

		void CopyActivities	
		( 
			const SpatialLayeredNet<Implementation>&   
		);

		void ReverseActivities
		( 
			const SpatialLayeredNet<Implementation>&   
		);

		void CalculateCovariance
		(
			const SpatialLayeredNet<Implementation>&   
		);

		bool CheckArchitecture
		( 
			const SpatialLayeredNet<Implementation>&   
		) const;

		bool CheckReverseArchitecture
		(
			const SpatialLayeredNet<Implementation>&   
		) const;

		//! From a physical position in a net, the corresponding position in a 
		//! reversed version of the net is computed.
		void ReverseNetPosition
		(
			PhysicalPosition&
		) const; 

		//! The z-positions of the net are reversed
		void ReverseAllNetPositions();

		//! scale all weights by a common factor
		void ScaleWeights(double);


		//! scale all weights to a given layer by a common factor
		void ScaleWeights(Layer, double);


	protected:

		virtual string FileHeader() const;
		virtual string FileFooter() const;
	
	private:


	   LayeredArchitecture*	pArchitecture  (const AbstractLinkRelation*);

	   istream& RemoveHeader(istream&);
	   istream& RemoveFooter(istream&);

	   LayeredArchitecture*	_p_dummy;

	   // already necessary for initialization base class, and expensive, hence auto_ptr
	   auto_ptr<NodeIdPosition>		_p_position_id; 
	  
	}; // end of SpatialLayeredNet


	typedef SpatialLayeredNet<D_LayeredReversibleSparseImplementation> SpatialConnectionistNet;
	typedef NodeIterator<SpatialConnectionistNet::NodeType> SCNodeIterator;
	typedef SpatialConnectionistNet::NodeType::predecessor_iterator SCPDIterator;
	typedef BackpropTrainingVector
		   <
			 D_LayeredReversibleSparseImplementation,
			 D_LayeredReversibleSparseImplementation::WeightLayerIterator
		   >      SpatialConnectionistTrainingVector;
	typedef BackpropTrainingVector
		   <
			D_LayeredReversibleSparseImplementation,
			D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold
		   >      SpatialConnectionistThresholdTrainingVector;

	typedef Hebbian<D_LayeredReversibleSparseImplementation> SpatialConnectionistHebbian;

} // end of Strucnet

#endif
