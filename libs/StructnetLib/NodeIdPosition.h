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
#ifndef _CODE_LIBS_STRUCNET_NODEIDPOSITION_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_NODEIDPOSITION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <utility>
#include "../ConnectionismLib/ConnectionismLib.h"
#include "ForwardOrder.h"
#include "ReverseOrder.h"
#include "RZOrder.h"
#include "LinkRelation.h"
#include "PhysicalPosition.h"
#include "LayerDescription.h"

using std::pair;
using NetLib::Layer;
using NetLib::NodeId;
using NetLib::NodeLinkCollection;

namespace StructnetLib
{
	

	//! A LayerDescription constains a full description of a spatially organised feedfoward network.
	//! The Layers are typically two dimensional (see LayerDecsription). each Layer describes one layer 
	//! in the network, so the complete feedfoward network is determined by a vector<LayerDescription>.
	//! It is then possible to assign a 1-1 relation between spatial position and NodeId, which is what
	//! NodeIdPosition does.	

	class NodeIdPosition 
	{
	public:

		typedef pair<NodeId, PhysicalPosition> NodeIdPositionPair;


		//! read a NodeIdPosition from a stream
		NodeIdPosition
		(
			istream&
		);

		//! Typically NodeIdPosition is constructed from a vector of LayerDescription
		NodeIdPosition
		(
			const vector<LayerDescription>&
		);

		//! Copy constructor
		NodeIdPosition
		(
			const NodeIdPosition&
		);

		//! Copy operator
		NodeIdPosition& operator=(const NodeIdPosition&);

		//! From NodeId to physical position 
		const PhysicalPosition&	Position(NodeId) const;

		//! From PhysicalPosition to NodeId
		NodeId	Id(const PhysicalPosition&)	const;

		//! WARNING! WILL BECOME OBSOLETE
		//! Used when creating feedback networks. Without this call the input
		//! layer of a feedback network will correspond to the output nodes of
		//! a feedforward network. 
		void ReverseZPositions();

		//! Gives an order for positions, to be used in iterators. It is NOT an iterator
		//! because there is no object to point to. NodeIdPosition is an auxiliary object,
		//! which may be part of a container (such as a network), and unless the container
		//! is known, it is not clear what the iterator should point to. ForwardOrder
		//! Orders positions such that after each iteration SpatialPosition_new > SpatialPosition_old.
		//! The ordering of SpatialPosition is defined in its header file.

		ForwardOrder begin() const;

		ForwardOrder end() const ;

		//! ReverseOrder  starts with the largest SpatialPosition (ordering defined
		//! in the header file), and counts down to the lowest.
		ReverseOrder rbegin() const;

		ReverseOrder rend() const;

		//! RZOrder starts with the smallest SpatialPosition of a network that is reversed
		//! in the z-direction. This is important in feedback networks that are generated
		//! from feedfoward networks. The smallest SpatialPosition in such a network would
		//! start at its input layer, which is the output layer of the original network. In
		//! order to iterate through feedfoward and feedback network in a similar fashion,
		//! use RZOrder

		RZOrder rzbegin() const;

		RZOrder rzend() const;

		//! endow a LinkRelation with a NodeIdPositon to obtain a NodeLinkCollection
		NodeLinkCollection	 Collection
		(
			const AbstractLinkRelation&
		) const;

		//! Produce a vector of Layers
		const vector<Layer>& VectorLayer			()	const;

		//! Reproduce the LayerDescription that first created this NodeIdPosition
		const vector<LayerDescription>&	Dimensions	()	const;

		friend ostream& operator<<
		(
			ostream&, 
			const NodeIdPosition& 
		);
		
		friend istream& operator>>
		(
			istream&,       
			NodeIdPosition& 
		);

	private:

		void ReverseZPosition();

		vector<LayerDescription>	_vec_desc_data;
		vector<NodeIdPositionPair>	_vec_id_position;
		vector<Layer>				_vec_desc;		

	}; // end of NeuronIdPosition


	ostream& operator<<( std::ostream&, const NodeIdPosition& );
	istream& operator>>( std::istream&,       NodeIdPosition& );

} // end of Strucnet


#endif // include guard
