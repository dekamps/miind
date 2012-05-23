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
#ifndef _CODE_LIBS_CONNECTIONISM_LAYEREDNETWORK_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISM_LAYEREDNETWORK_INCLUDE_GUARD


#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include <iostream>
#include <string>
#include <memory>

#include "../NetLib/NetLib.h"
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "../UtilLib/UtilLib.h"
#include "BasicDefinitions.h"
#include "TrainingAlgorithm.h"
#include "TrainingUnit.h"


using std::auto_ptr;
using NetLib::Pattern;
using NetLib::NodeId;
using NetLib::ConstNodeIterator;
using NetLib::Sigmoid;
using NetLib::SigmoidParameter;
using SparseImplementationLib::SparseImplementation;
using SparseImplementationLib::D_SparseImplementation;
using UtilLib::Index;
using UtilLib::Number;

// Created:				26-07-1999
// Author:				Marc de Kamps

namespace ConnectionismLib {

  //! LayeredNetwork

	template <class _Implementation>
    class LayeredNetwork 
    {
    public:

		typedef TrainingAlgorithm<_Implementation>	 training;
		typedef TrainingAlgorithm<_Implementation>& training_reference;
		typedef TrainingAlgorithm<_Implementation>* training_pointer;

		typedef AbstractSquashingFunction*         squashing_pointer;
		typedef const AbstractSquashingFunction&   const_squashing_reference;

		typedef _Implementation Implementation;
		typedef typename _Implementation::NodeType NodeType;
		// ctors, dtors and the like:
				
		//! Create network from an input stream
		LayeredNetwork
		(
			istream&
		);

		//! Create from a LayeredArchitecture
		LayeredNetwork
		(
		        NetLib::LayeredArchitecture*
		 );

		//! Overrride default squashing function
		LayeredNetwork
		(
		        NetLib::LayeredArchitecture*, 
				const_squashing_reference
		);

		//! Use default squashing function (Sigmoid), but override default sigmoid parameter
		LayeredNetwork
		(
		        NetLib::LayeredArchitecture*, 
				const SigmoidParameter&
		);

		//! Copy constructor
		LayeredNetwork
		(
				const LayeredNetwork&
		); 

		//! anticipate derived classes: make destructor virtual
		 virtual ~LayeredNetwork();

		//! copy operator (the TrainingAlgorithm object is not copied, the new network must be configured(
		//! with a new TrainingAlgorithm (if this turns out to be inconvenient in the future, it may be changed
		LayeredNetwork& operator=(const LayeredNetwork&);

		// streaming operator

		virtual bool ToStream(ostream&) const;

		//! Adding training algorithm
		void SetTraining (training_reference);

		// important network functions:

		//! Evolve the Network in an Order, prescribed by Order 
		virtual bool  Evolve();

		// Input and out functions

		//! Enter pattern in Input Nodes
		 bool ReadIn(const Pattern<typename Implementation::NodeValue>&);

		//! Read pattern from output Nodes
		Pattern<typename Implementation::NodeValue> ReadOut() const;

		//! Read out individual Node
		double GetActivity (NodeId) const;

		//! Setvalue of indivdual Node
		void SetActivity(NodeId, double);

		//! If you really need it (implementation depenedent, can be very slow): normal access is via TraningAlgorithm
		 double GetWeight(NodeId, NodeId) const;

		// Training
		bool Train( const TrainingUnit<typename Implementation::NodeValue>& );
 
		bool Initialize();

		ConstNodeIterator<typename _Implementation::NodeType> begin() const;

		ConstNodeIterator<typename _Implementation::NodeType> end() const;

		NodeIterator<typename _Implementation::NodeType> begin();

		NodeIterator<typename _Implementation::NodeType> end();

		NodeIterator<typename _Implementation::NodeType> begin(Layer);

		NodeIterator<typename _Implementation::NodeType> end(Layer);

		// Network properties (will be deprecated)
	
		double MaxActivation() const;
		double MinActivation() const;

		Number NumberOfInputNodes () const;
		Number NumberOfOutputNodes() const;
		Number NumberOfNodes      () const;
		Number NumberOfLayers     ()  const;
		Number NumberOfNodesInLayer(Layer) const;

		NodeId BeginId(Layer) const;

		//! this is the last NodeId of a given layer, it refers to an existing node, and not to one past the last existing node, as in STL-style end
		NodeId EndId  (Layer) const;

		bool IsInputNeuronFrom(NodeId, NodeId) const;


		template <class I> friend ostream& operator<<(ostream&, const LayeredNetwork<I>&);
		template <class I> friend istream& operator>>(istream&,       LayeredNetwork<I>&);

    protected:

		Implementation& Imp(){ return _implementation;}

		virtual string FileHeader() const;
		virtual string FileFooter() const;

    private:

		istream& RemoveHeader (istream&) const;
		istream& RemoveFooter (istream&) const;

		Implementation _implementation;

		auto_ptr<AbstractSquashingFunction> ImportSquashingFunction(istream& s) const;
		training_pointer		_p_train;

    }; // end of LayeredNetwork

  template <class Implementation>
  ostream& operator<<(ostream& , const LayeredNetwork<Implementation>&);

  template <class Implementation>
  istream& operator>>(istream& ,	LayeredNetwork<Implementation>&);
														

  typedef LayeredNetwork<D_SparseImplementation > SparseLayeredNetwork;

  typedef NodeIterator<D_SparseImplementation::NodeType> D_NodeIterator;

} // end of Connectionism

#endif // include guard
