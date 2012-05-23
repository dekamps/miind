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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_STRUCNET_BIONETCODE_INCLUDE_GUARD
#define _CODE_LIBS_STRUCNET_BIONETCODE_INCLUDE_GUARD


#include "SparseLayeredNet.h"
#include "BasicDefinitions.h"
#include "LocalDefinitions.h"

namespace StructnetLib
{

	template <class Implementation>
	SpatialLayeredNet<Implementation>::SpatialLayeredNet
	(
		const AbstractLinkRelation* p_link_relation
	):
	LayeredNetwork<Implementation>
	(
		pArchitecture(p_link_relation)
	),
	_p_position_id(new NodeIdPosition(p_link_relation->VectorLayerDescription()))
	{
		delete _p_dummy;
		_p_dummy = 0;
	}

	template <class Implementation>
	SpatialLayeredNet<Implementation>::SpatialLayeredNet
	(
		const AbstractLinkRelation* p_link_relation,
		const SigmoidParameter& param
	):
	LayeredNetwork<Implementation>
	(
		pArchitecture(p_link_relation),
		param
	),
	_p_position_id(new NodeIdPosition(p_link_relation->VectorLayerDescription()))
	{
		delete _p_dummy;
		_p_dummy = 0;
	}

	template <class Implementation>
	SpatialLayeredNet<Implementation>::SpatialLayeredNet
	(
		const SpatialLayeredNet<Implementation>& rhs
	):
	LayeredNetwork<Implementation>
	(
		rhs
	),
	_p_position_id(new NodeIdPosition(*rhs._p_position_id))
	{
	}
	template <class Implementation>
	SpatialLayeredNet<Implementation>::SpatialLayeredNet
		(
			const AbstractLinkRelation*			p_link_relation,
			const AbstractSquashingFunction&	function
		):
	LayeredNetwork<Implementation>
		(
			pArchitecture(p_link_relation),
			function
		),
	_p_position_id(new NodeIdPosition(p_link_relation->VectorLayerDescription()))
	{
		delete _p_dummy;
		_p_dummy = 0;
	}

	template <class Implementation>
	SpatialLayeredNet<Implementation>::~SpatialLayeredNet()
	{ 
	}


	template <class Implementation>	
	istream& SpatialLayeredNet<Implementation>::RemoveHeader(istream& s)
	{
		// Remove the BioNet header
		string str_header;
		s >> str_header;

		if (str_header != SpatialLayeredNet<Implementation>::FileHeader())
			throw NetworkParsingException(string("BioNet file header expected"));

		return s;
	}

	template <class Implementation>	
	istream& SpatialLayeredNet<Implementation>::RemoveFooter(istream& s)
	{
		// Remove the BioNet header
		string str_header;
		s >> str_header;

		if (str_header != SpatialLayeredNet<Implementation>::FileFooter())
			throw NetworkParsingException(string("SpatialLayeredNet file footer expected"));

		return s;
	}

	template <class Implementation>
	SpatialLayeredNet<Implementation>::SpatialLayeredNet(istream& s):
	LayeredNetwork<Implementation>(RemoveHeader(s)),
	_p_position_id(new NodeIdPosition(s))
	{
		RemoveFooter(s);
	}
	template <class Implementation>
	inline const PhysicalPosition& SpatialLayeredNet<Implementation>::Position
	( 
		NodeId nid
	) const
	{
		return _p_position_id->Position(nid);
	}

	template <class Implementation>
	inline NodeId SpatialLayeredNet<Implementation>::Id
	(
		const PhysicalPosition& ls
	) const
	{
		return _p_position_id->Id(ls);
	}

	template <class Implementation>
	const vector<LayerDescription>&  SpatialLayeredNet<Implementation>::Dimensions() const
	{
		return _p_position_id->Dimensions();
	}

	/// still needed by netview. TODO: phase out
	template <class Implementation>
	void SpatialLayeredNet<Implementation>::ReverseAllNetPositions()

	{
		_p_position_id->ReverseZPositions();
	}

	template <class Implementation> 
	bool SpatialLayeredNet<Implementation>::CheckReverseArchitecture
	( 
		const SpatialLayeredNet& reverse_net
	) const
	{
		Index n_layer_index;
		Layer n_layers = this->NumberOfLayers() - 1;

		for ( n_layer_index = 0; n_layer_index < n_layers; n_layer_index++ )
		{
			LayerDescription forward = this->Dimensions()[n_layer_index];
			LayerDescription reverse = reverse_net.Dimensions()[n_layers - n_layer_index];

			if ( forward != reverse )
				return false;
		}

		return true;
	}
		
	template <class Implementation> 
	bool SpatialLayeredNet<Implementation>::CheckArchitecture
	( 
		const SpatialLayeredNet& net
	) const	
	{
		Layer n_layers = this->NumberOfLayers() - 1;

		for (Index n_layer_index = 0; n_layer_index < n_layers; n_layer_index++ )
		{
			LayerDescription forward_this =     Dimensions()[n_layer_index];
			LayerDescription forward_net  = net.Dimensions()[n_layer_index];

			if ( forward_this != forward_net )
				return false;
		}
			return true;
	}

	template <class Implementation>
	void SpatialLayeredNet<Implementation>::ReverseActivities
	( 
		const SpatialLayeredNet<Implementation>& net_reverse 
	)
	{
		assert ( CheckReverseArchitecture(net_reverse) );
		assert (net_reverse.NumberOfLayers() > 0 );

		Number nr_neurons_reverse = net_reverse.NumberOfNodes();

		// loop over all neurons
		for (Index  n_id_value = 1; n_id_value <= nr_neurons_reverse; n_id_value++ )
		{
			NodeId id(n_id_value);

			//  the position of the id in this network
			PhysicalPosition current_struc = Position(id);

			// calculate the corrsponding position in the reverse network
			ReverseNetPosition(current_struc);
		
			// get the id of the corresponding network in the reverse network
			NodeId IdReverse = net_reverse.Id(current_struc);

			// set its activity in this network
			SetActivity(id,net_reverse.GetActivity(IdReverse) );
		}			
	}

	template <class Implementation>
	void SpatialLayeredNet<Implementation>::CalculateCovariance
	( 
		const SpatialLayeredNet<Implementation>& net_reverse 
	)
	{
		assert ( CheckReverseArchitecture(net_reverse) );
	
		Number nr_neurons_reverse = net_reverse.NumberOfNodes();

		// loop over all neurons
		for (Index n_id_value = 1; n_id_value <= nr_neurons_reverse; n_id_value++ )
		{
			NodeId id(n_id_value);

			//  the position of the id in this network
			PhysicalPosition current_struc = Position(id);

			// calculate the corrsponding position in the reverse network
			ReverseNetPosition(current_struc);

			// get the id of the corresponding network in the reverse network
			NodeId IdReverse = net_reverse.Id(current_struc);

			// calculate the covariance
			double f_covariance = this->GetActivity(id)*net_reverse.GetActivity(IdReverse);

			// set the covariance in the network
			this->SetActivity(id, f_covariance );
		}			
	}

	template <class Implementation>
	void SpatialLayeredNet<Implementation>::CopyActivities
	( 
		const SpatialLayeredNet<Implementation>& net 
	)
	{
		assert ( CheckArchitecture(net) );
	
		Number nr_neurons = net.NumberOfNodes();
		Layer n_max = net.NumberOfLayers() - 1;

		// loop over all neurons
		for (Index n_id_value = 1; n_id_value <= nr_neurons; n_id_value++ )
		{
			NodeId id(n_id_value);

			// set its activity in this network
			SetActivity(id,net.GetActivity(id) );
		}			
	}

	template <class Implementation>
	string SpatialLayeredNet<Implementation>::FileHeader() const
	{
		return STR_BIONET_HEADER;
	}

	template <class Implementation>
	string SpatialLayeredNet<Implementation>::FileFooter() const
	{
		return STR_BIONET_FOOTER;
	}

	template <class Implementation>
	bool SpatialLayeredNet<Implementation>::ToStream(ostream& s) const
	{	
		s << SpatialLayeredNet<Implementation>::FileHeader() << "\n";

		// Also get the base class on the stream,
		LayeredNetwork<Implementation>::ToStream(s);
		s << *_p_position_id;
			
		s << SpatialLayeredNet<Implementation>::FileFooter() << "\n";

		return true;
	}


	template <class Implementation>
	LayeredArchitecture* SpatialLayeredNet<Implementation>::pArchitecture
	(
		const AbstractLinkRelation* p_link_relation
	) 
	{	
		vector<LayerDescription> vector_description = p_link_relation->VectorLayerDescription();
		NodeIdPosition id_position(vector_description);

		NodeLinkCollection* _p_collection = 
			new NodeLinkCollection
			(
				id_position.Collection(*p_link_relation)
			);

		vector<Layer> vector_layer = id_position.VectorLayer();

		// ugly, but necessary, ANNetwork needs a pointer to a
		// LayeredArchitecture object, which thereafter has to be destroyed.
		// It is associated with _p_dummy to retain a  reference to the object
		// to allow destruction lateron

		_p_dummy = 
			new LayeredArchitecture
			(
				vector_layer, 
				_p_collection
			);

		return _p_dummy;
	}

	template <class Implementation> 
	void SpatialLayeredNet<Implementation>::ReverseNetPosition(PhysicalPosition& ls) const
	{
		assert(this->NumberOfLayers() - 1  >= ls._position_z);
		Number n_max = this->NumberOfLayers() - 1;
		ls._position_z = n_max - ls._position_z;
	}


	template <class Implementation>
	void SpatialLayeredNet<Implementation>::ScaleWeights(double scale)
	{
		this->Imp().ScaleWeights(scale);
	}

	template <class Implementation>
	void SpatialLayeredNet<Implementation>::ScaleWeights(Layer l, double scale)
	{
		assert( l > 0 && l < this->NumberOfLayers() );
		for (SparseImplementationLib::RDLNodeIterator iter = this->begin(l); iter != this->end(l); iter++)
			iter->ScaleWeights(scale);
	}

} // end of Strucnet


#endif // include guard
