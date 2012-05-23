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
#ifndef _CODE_LIBS_CONNECTIONISM_LAYEREDNETWORKCODE_INCLUDE_GUARD
#define _CODE_LIBS_CONNECTIONISM_LAYEREDNETWORKCODE_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "LayeredNetwork.h"


namespace ConnectionismLib {

	template <class Implementation>
	LayeredNetwork<Implementation>::~LayeredNetwork() 
	{
		delete _p_train;
	}

	template <class Implementation>
	LayeredNetwork<Implementation>::LayeredNetwork
	(
		LayeredArchitecture* p_architecture
	):
	_implementation(p_architecture),
	_p_train(0)
	{

		Sigmoid sigmoid;
		_implementation.ExchangeSquashingFunction(sigmoid);
	}
		
	template <class Implementation>
    LayeredNetwork<Implementation>::LayeredNetwork
    (
		LayeredArchitecture* p_architecture, 
		const SigmoidParameter& parameter_sigmoid
    ):
    _implementation(p_architecture),
    _p_train(0)
	{
		Sigmoid sigmoid(parameter_sigmoid);
		_implementation.ExchangeSquashingFunction(sigmoid);
	}

  template <class Implementation>
    LayeredNetwork<Implementation>::LayeredNetwork
    (
		LayeredArchitecture* p_architecture, 
		const AbstractSquashingFunction&	function	
     ):
    _implementation(p_architecture),
    _p_train(0) 
	{
		_implementation.ExchangeSquashingFunction(function);
    }

	template <class Implementation>
	LayeredNetwork<Implementation>::LayeredNetwork
    (
		const LayeredNetwork<Implementation>& rhs
     ):
    _implementation(rhs._implementation),
	_p_train( (rhs._p_train != 0 ) ? rhs._p_train->Clone(&_implementation) : 0) 
	{
    }
		
	template <class Implementation>
    LayeredNetwork<Implementation>::LayeredNetwork(istream& s):
    _implementation(RemoveHeader(s)),
    _p_train(0)
	{
		RemoveFooter(s);
    }
		
	template <class Implementation>
    LayeredNetwork<Implementation>&  LayeredNetwork<Implementation>::operator=
    (
	     const LayeredNetwork<Implementation>& rhs
    )    
	{
		if ( this == &rhs )
			return *this;

		_implementation = rhs._implementation;
       
		// training algorithm is not copied
		delete _p_train;
		_p_train = 0;

		return *this;
    }

	template <class Implementation>
    Number LayeredNetwork<Implementation>::NumberOfInputNodes() const 
	{
		return _implementation.NumberOfInputNodes();
    }

	template <class Implementation>
    Number LayeredNetwork<Implementation>::NumberOfOutputNodes() const
    {
      return _implementation.NumberOfOutputNodes();
    }

	template <class Implementation>
    Number LayeredNetwork<Implementation>::NumberOfNodesInLayer(Layer n_layer) const 
	{
		return _implementation.NumberOfNodesInLayer(n_layer);
    }

	template <class Implementation>
    NodeId LayeredNetwork<Implementation>::BeginId(Layer n_layer) const 
	{
		return _implementation.BeginId(n_layer);
    }

	template <class Implementation>
    NodeId LayeredNetwork<Implementation>::EndId(Layer n_layer) const 
	{
		return NodeId(_implementation.BeginId(n_layer)._id_value + this->NumberOfNodesInLayer(n_layer)-1);
	}

	template <class Implementation>
	bool LayeredNetwork<Implementation>::IsInputNeuronFrom
	(
		NodeId In, 
		NodeId Out
	) const 
	{
		int dif = Out._id_value - 1;
		ConstNodeIterator<typename Implementation::NodeType> node_out = _implementation.begin() + dif;
		
		typename Implementation::NodeType* p_node_out = const_cast<typename Implementation::NodeType*>( node_out.operator->() );

		NodeId id = p_node_out->MyNodeId();
		
		typename Implementation::NodeType::predecessor_iterator iter =
			find
			(
				p_node_out->begin(),
				p_node_out->end(),
				In
			);
		return (iter != p_node_out->end());
	}

	template <class Implementation>
	double LayeredNetwork<Implementation>::GetWeight(NodeId id_in, NodeId id_out) const 
	{
		double ret = 0.0;	
		_implementation.GetWeight(id_in,id_out,ret);
		return ret;
	}

	template <class Implementation>
	Pattern<typename Implementation::NodeValue>  LayeredNetwork<Implementation>::ReadOut() const 
	{
		return _implementation.ReadOut();
	}

	template <class Implementation>
	bool  LayeredNetwork<Implementation>::Train
	( 
		const TrainingUnit<typename Implementation::NodeValue>& tu 
	) 
	{
		return _p_train->Train(tu);
	}

	template <class Implementation>
	bool  LayeredNetwork<Implementation>::Initialize()
	{
		return _p_train->Initialize();
	}

  template <class Implementation>
    void LayeredNetwork<Implementation>::SetActivity(NodeId nid, double val)
    {
      _implementation.Insert(nid,val);
    }

	template <class Implementation>
	double LayeredNetwork<Implementation>::GetActivity(NodeId nid) const
	{
		return _implementation.Retrieve(nid);
	}

	template <class Implementation>
	bool  LayeredNetwork<Implementation>::Evolve() 
	{
		for 
		(
			typename Implementation::Order iter = _implementation.begin() + NumberOfInputNodes();
			iter != _implementation.end();
			iter++
		)
			iter->Update();

		return true;
	}

	template <class Implementation>
		NodeIterator<typename Implementation::NodeType>
	LayeredNetwork<Implementation>::begin()
	{
		return _implementation.begin();
	}

	template <class Implementation>
		NodeIterator<typename Implementation::NodeType>
	LayeredNetwork<Implementation>::end() 
	{
		return _implementation.end();
	}

	template <class Implementation>
		NodeIterator<typename Implementation::NodeType>
	LayeredNetwork<Implementation>::begin(Layer l)
	{
		assert(l <= this->NumberOfLayers() );
		NodeId id = this->BeginId(l);
		// begin starts at node 1!, so whatever the begin id it should not be zero
		assert(id._id_value > 0);
		// so the subtraction is always well defined
		return _implementation.begin() + (id._id_value - 1);
	}

	template <class Implementation>
		NodeIterator<typename Implementation::NodeType>
	LayeredNetwork<Implementation>::end(Layer l) 
	{
		assert(l < this->NumberOfLayers() );
		NodeId id = this->EndId(l);
		// not + 1, because begin() already starts at NodeId 1
		return _implementation.begin() + id._id_value;
	}

	template <class Implementation>
		ConstNodeIterator<typename Implementation::NodeType>
	LayeredNetwork<Implementation>::begin() const
	{
		return _implementation.begin();
	}

	template <class Implementation>
		ConstNodeIterator<typename Implementation::NodeType>
	LayeredNetwork<Implementation>::end() const
	{
		return _implementation.end();
	}

	template <class Implementation>
	bool LayeredNetwork<Implementation>::ReadIn( const Pattern<typename Implementation::NodeValue>& pat )
	{
		return _implementation.ReadIn(pat);
	}

  template <class Implementation>
    void LayeredNetwork<Implementation>::SetTraining
	( 
		TrainingAlgorithm<Implementation>& alg 
	) {
      _p_train = alg.Clone(&_implementation);
    }


  template <class Implementation>
    Number LayeredNetwork<Implementation>::NumberOfNodes() const {
		return _implementation.NumberOfNodes();
    }

  template <class Implementation>	
    Number  LayeredNetwork<Implementation>::NumberOfLayers() const {
      return _implementation.NumberOfLayers();
    }
	
	namespace {

		template<class Implementation>
		bool MaxActivationNode
		(
			typename Implementation::NodeType node_1, 
			typename Implementation::NodeType node_2
		)
		{
			return (node_1.SquashingFunction().MaximumActivity() < node_2.SquashingFunction().MaximumActivity());
		}
	}
	template <class Implementation>
	double  LayeredNetwork<Implementation>::MaxActivation() const 
	{	
		typename Implementation::NodeType* p_begin = const_cast<typename Implementation::NodeType*>( _implementation.begin().operator->());
		typename Implementation::NodeType* p_end   = const_cast<typename Implementation::NodeType*>( _implementation.end().operator->());

		typename Implementation::NodeType* p_max =
			std::max_element
			(
				p_begin,
				p_end,
				MaxActivationNode<Implementation>
			);

		return p_max->SquashingFunction().MaximumActivity();
	}

	namespace {
		template<class Implementation>
		bool MinActivationNode
		(
			typename Implementation::NodeType node_1, 
			typename Implementation::NodeType node_2
		)
		{
			return (node_1.SquashingFunction().MinimumActivity() < node_2.SquashingFunction().MinimumActivity());
		}
	}
	template <class Implementation>
	double  LayeredNetwork<Implementation>::MinActivation() const 
	{
		typename Implementation::NodeType* p_begin = const_cast<typename Implementation::NodeType*>( _implementation.begin().operator->());
		typename Implementation::NodeType* p_end   = const_cast<typename Implementation::NodeType*>( _implementation.end().operator->());

		typename Implementation::NodeType* p_min =
			std::min_element
			(
				p_begin,
				p_end,
				MinActivationNode<Implementation>
			);

		return p_min->SquashingFunction().MinimumActivity();

	}
  

  template <class Implementation>	
    string  LayeredNetwork<Implementation>::FileHeader() const {
      return STR_NETWORK_HEADER;
    }

	template <class Implementation>	
	string  LayeredNetwork<Implementation>::FileFooter() const 
	{
		return STR_NETWORK_FOOTER;
	}

	template <class Implementation>
	istream& LayeredNetwork<Implementation>::RemoveFooter(istream& s) const 
	{
		string str_footer;
		s >> str_footer;

		return s;
	}

	template <class Implementation>
	istream& LayeredNetwork<Implementation>::RemoveHeader(istream& s) const 
	{
		string str_header;
		// and the Network header
		s >> str_header;
		if (str_header != LayeredNetwork<Implementation>::FileHeader())
			throw NetworkParsingException(string("Network file header expected"));

		return s;
	}


	template <class Implementation>
	auto_ptr<AbstractSquashingFunction> LayeredNetwork<Implementation>::ImportSquashingFunction(istream& s) const 
	{
		SquashingFunctionFactory factory;
		return factory.FromStream(s);
	}

  template <class Implementation>
    istream& operator>>(istream& s, LayeredNetwork<Implementation>& Net){
      s >> Net._implementation;
      return s;
    }

  template <class Implementation>
    bool LayeredNetwork<Implementation>::ToStream(ostream& s) const {

      s << LayeredNetwork<Implementation>::FileHeader() << "\n";
      s << _implementation;
      s << LayeredNetwork<Implementation>::FileFooter() << "\n";

      return true;
    }

  template <class Implementation>
    ostream& operator<<(ostream& s, const LayeredNetwork<Implementation>& Net){
      Net.ToStream(s);
      return s;
    }

} // end of Connectionism

#endif // include guard
