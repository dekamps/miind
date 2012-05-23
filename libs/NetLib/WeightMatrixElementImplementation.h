// Copyright (c) 2005 - 2008 Marc de Kamps
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

#ifndef _CODE_LIBS_NETLIB_WMEIMPLEMENTATION_INCLUDE_GUARD
#define _CODE_LIBS_NETLIB_WMEIMPLEMENTATION_INCLUDE_GUARD

#ifdef WIN32
#pragma warning(disable: 4786)
#endif

#include "../NumtoolsLib/NumtoolsLib.h"
#include "../UtilLib/UtilLib.h"
#include "BasicDefinitions.h"
#include "LayeredImplementation.h"
/*
using NetLib::Pattern_ifc;
using Libs::Numtools::QaDirty;


	namespace ImplementationLib
	{

	struct ElementId {
		explicit ElementId( int val ):value(val){}

		int value;
	};


	template <class NodeValue, class WeightValue> class LayerElement;

	template <class NodeValue, class WeightValue>
	class WeightMatrixElementImplementation : public LayeredImplementation
	{

public:

	typedef WeightMatrixElementImplementation<NodeValue,WeightValue>&		implem_ref;
	typedef const WeightMatrixElementImplementation<NodeValue,WeightValue>&	const_implem_ref;
	typedef LayerElement<NodeValue,WeightValue>					element;
	typedef Pattern_ifc<NodeValue>								pattern;
	typedef std::ostream&											ostream_ref;
//	typedef LyrWeightIterIO<WeightMatrixElementImplementation<NeuronValue,WeightValue> >&                 lyrweight_iter_ref;
//	typedef const LyrWeightIterIO<WeightMatrixElementImplementation<NeuronValue,WeightValue> >&           const_lyrweight_iter_ref;

				WeightMatrixElementImplementation( Architecture& Arch  );
				WeightMatrixElementImplementation( const WeightMatrixElementImplementation & );
				~WeightMatrixElementImplementation();
	implem_ref	operator=( const_implem_ref );

	double		Update(NodeId) const;
	void		Insert( NodeId, NodeValue );
	NodeValue Retrieve        (      NodeId              ) const;
//	void		InsertWeight(const_lyrweight_iter_ref, WeightValue);
//	WeightValue GetWeight( const_lyrweight_iter_ref) const;
//	NodeId	NeuronInId( const_lyrweight_iter_ref ) const;
//	NodeId	NeuronOutId( const_lyrweight_iter_ref ) const;
	void		InsertWeight( NodeId, NodeId, WeightValue);
	WeightValue GetWeight( NodeId, NodeId ) const;

	bool		ReadIn( const Pattern_ifc<NodeValue>& );
	pattern		ReadOut() const;

	virtual bool ToStream(ostream& s);

	//TODO: these function should now be handled by the base class CHECK !!
	// Network property functions:
	size_t NumberOfNodes()		const;
//	size_t NrInput()		const;
//	size_t NrOutput()		const;
//	size_t NrConnections()	const;
//	size_t NrLayers()		const;
//	size_t NrLargestLayer() const;
//	size_t  NrConnectionFrom(size_t)const;
//	size_t NrNeuronsInLayer(Layer) const;
//	NodeId BeginId(Layer) const;

private:

	std::vector<element*>	InitializeElementVector		( Architecture& );
	std::vector<size_t>		InitializeAccumulativeVector( Architecture& );

	ElementId				WhichNeuronInElement		( NodeId ) const;
	size_t					WhichElementInVector		( NodeId ) const;
	ElementId				WhichNeuronToUpdate			( NodeId ) const;
	size_t					WhichElementToUpdate		( NodeId ) const;
	ElementId				InputIndex					(Index, Layer) const;
	ElementId				OutputIndex					(Index, Layer) const;
	bool					IsLastInputNeuron			( size_t, int ) const;
	bool					IsLastOutputNeuron			( size_t, int ) const;

	NodeId				WhichNeuronInput			( size_t, ElementId ) const;
	NodeId				WhichNeuronOutput			( size_t, ElementId ) const;
	struct WeightReference{

		size_t		_element;
		ElementId	_in;
		ElementId	_out;

		WeightReference():_element(0),_in( ElementId(0) ), _out( ElementId(0) ){}
	};

	WeightReference			FromIdToWeightReference		(NodeId, NodeId) const;

//	LayerStructure			_structure_layer;
	std::vector<element*>	_vec_element;
	std::vector<Layer>		_vec_accumulative_neurons;

};


template <class NeuronValue, class WeightValue>
class LayerElement {
public:

	typedef LayerElement& ref_element;

				LayerElement( size_t, size_t );
				LayerElement( const LayerElement& );
			   ~LayerElement();

	ref_element operator=( const LayerElement&  );

	double		Update			( ElementId, NodeId   )		const;
	void		Insert			( ElementId,NeuronValue );
	NeuronValue Retrieve		( ElementId				)		const;
	WeightValue RetrieveWeight	( ElementId, ElementId  )		const;
	void InsertWeight			( ElementId, ElementId, WeightValue );
	Number NrInput() const;
	Number NrOutput() const;

private:

	std::vector<NeuronValue> _vec_neur;
	QaDirty<WeightValue >	 _mat_weights;
};

	}; // end of NetLib

*/
#endif //include guard


