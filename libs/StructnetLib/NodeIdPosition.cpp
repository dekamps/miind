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

#include "NodeIdPosition.h"
#include "OrientedPatternCode.h"
#include "InconsistentStructureException.h"
#include "NeuronIdPositionParsingException.h"
#include "LocalDefinitions.h"

#include <iostream>

using std::cout;
using std::endl;
using namespace NetLib;
using namespace UtilLib;
using namespace StructnetLib;

NodeIdPosition::NodeIdPosition
(
	const vector<LayerDescription>& vec_struc
):
_vec_desc_data(vec_struc),
_vec_desc(vec_struc.size())
{
	// Tested: 11-12-1999
	// Author: Marc de Kamps


	Index nr_id = 1;
	Index nr_start_of_layer_id = 1;

	std::pair<NodeId,PhysicalPosition>	current_pair;	
	PhysicalPosition			current_struct;


	for (Layer n_layer_index = 0; n_layer_index < vec_struc.size(); n_layer_index++ )
	{

		// calculate number neurons per layer
		Number nr_x_neur     = vec_struc[n_layer_index]._nr_x_pixels;
		Number nr_y_neur     = vec_struc[n_layer_index]._nr_y_pixels;
		Number nr_depth_neur = vec_struc[n_layer_index]._nr_features;
		_vec_desc[n_layer_index] = nr_x_neur*nr_y_neur*nr_depth_neur;

		// create an oriented pattern from which we read the Index,
		// which will serve as Id, hence there is automatic correspondence
		// between network layers and OrientedPattern's

		OrientedPattern<double> pat_ind(nr_x_neur,nr_y_neur,nr_depth_neur);

		// loop over all grid values
		for (Index n_depth = 0; n_depth < vec_struc[n_layer_index]._nr_features; n_depth++ )
			for (Index n_y = 0; n_y < vec_struc[n_layer_index]._nr_y_pixels; n_y++ )
				for (Index n_x = 0; n_x < vec_struc[n_layer_index]._nr_x_pixels; n_x++ )
				{
					// find grid parameter
					current_struct._position_x = n_x;
					current_struct._position_y = n_y;
					current_struct._position_z = n_layer_index;
					current_struct._position_depth = n_depth;
					
					// calculate corresponding neurond id
					current_pair.first  = NodeId( pat_ind.IndexFunction(n_x, n_y, n_depth ) + nr_start_of_layer_id );
					current_pair.second = current_struct;

					// push it back on vector
					_vec_id_position.push_back( current_pair );

					nr_id++;
				}
		nr_start_of_layer_id = nr_id;
	}

}

NodeIdPosition::NodeIdPosition(const NodeIdPosition& rhs):
_vec_desc_data(rhs._vec_desc_data),
_vec_id_position(rhs._vec_id_position),
_vec_desc(rhs._vec_desc)
{
}

const vector<Layer>& NodeIdPosition::VectorLayer() const
{
	// Tested: 11-12-1999
	// Author: Marc de Kamps
	return _vec_desc;
}

const PhysicalPosition& NodeIdPosition::Position(NodeId nid) const
{
	// Tested: 11-12-1999
	// Author: Marc de Kamps

	assert( nid._id_value <= static_cast<int>( _vec_id_position.size()) );
	return _vec_id_position[nid._id_value - 1].second;
}

NodeId NodeIdPosition::Id(const PhysicalPosition& ls) const
{
	// Dumb algorith; can be improved if necessary;
	// Tested: 11-12-1999
	// Author: Marc de Kamps

	Number nr_pos_pair = static_cast<Number>(_vec_id_position.size());

	for ( Index n_ind = 0; n_ind < nr_pos_pair; n_ind++ )
		if ( _vec_id_position[n_ind].second == ls )
			return _vec_id_position[n_ind].first; 

	throw InconsistentStructureException(string("Id should have retuned"));
}

const vector<LayerDescription>& NodeIdPosition::Dimensions() const
{
	return _vec_desc_data;
}

NodeLinkCollection NodeIdPosition::Collection (const AbstractLinkRelation& lr) const
{
	// Author: Marc de Kamps
	// Tested: 13-12-1999

	vector<NodeLink> vec_link;     // constructor argument for NeuronLinkCollection
	Number nr_neurons   = static_cast<Number>(_vec_id_position.size());

	for ( Index n_output_index = 0; n_output_index < nr_neurons; n_output_index++ )
	{
		vector<NodeId> vec_input;

		for( Index n_input_index = 0; n_input_index < nr_neurons; n_input_index++ )
		{
			if ( 
				lr
				( 
					_vec_id_position[n_input_index].second,
					_vec_id_position[n_output_index].second 
				) 
			)
				vec_input.push_back( _vec_id_position[n_input_index].first );

		}

		NodeId OutputId = _vec_id_position[n_output_index].first;
		NodeLink Link( OutputId, vec_input );
		vec_link.push_back( Link );
	}
	return NodeLinkCollection( vec_link );

}



ostream& StructnetLib::operator<<(ostream& s, const NodeIdPosition& nid_pos)

{

	s << "<NodeIdPosition>\n";

	// Write out the number of layers
	Index n_ind;
	s << static_cast<unsigned int>(nid_pos._vec_desc_data.size()) << "\n";

	// Write out the LayerStructureVector
	s << "<LayerStructureVector>\n";
	for ( n_ind = 0; n_ind < nid_pos._vec_desc_data.size(); n_ind++ )
		s << nid_pos._vec_desc_data[n_ind] << "\n";
	s << "</LayerStructureVector>\n";

	// Write out Id vs Positiom
	for ( n_ind = 0; n_ind < nid_pos._vec_id_position.size(); n_ind++ )
		s << nid_pos._vec_id_position[n_ind].first._id_value << "\t" 
		  << nid_pos._vec_id_position[n_ind].second          <<"\n";

	// Finish
	s << "</NodeIdPosition>\n";

	return s;
}

NodeIdPosition::NodeIdPosition( std::istream& s )
{
	string str_header;
	s >> str_header;

	if ( str_header!= "<NodeIdPosition>" )
		throw NeuronIdPositionParsingException(string("NodeIdPosition tag expected"));

	Index n_layers;
	s >> n_layers;

	// Input LayerStructure vector
	s >> str_header;
	if ( str_header != "<LayerStructureVector>" )
		throw NeuronIdPositionParsingException(string("(LayerStructureVector tag expected"));

	LayerDescription current_desc_data;

	Index n_ind;
	Index nr_nodes = 0;
	for ( n_ind = 0; n_ind < n_layers; n_ind++ )
	{
		s >> current_desc_data;
		_vec_desc_data.push_back( current_desc_data );

		// Calclulate total number of nodes
		nr_nodes += current_desc_data._nr_x_pixels*
					current_desc_data._nr_y_pixels*
					current_desc_data._nr_features;

	}

	string str_footer;
	s >> str_footer;
	if ( str_footer != "</LayerStructureVector>" )
		throw NeuronIdPositionParsingException(string("LayerStructureVector footer expected"));
	
	int n_id;

	_vec_id_position.resize(nr_nodes);
	for ( n_ind = 0; n_ind < nr_nodes; n_ind++ )
	{
		if( (n_ind+1) % NUMBER_OF_GENERATION_REPORT == 0 )
			cout << "\tReading in node " << (n_ind+1) << endl;
		NodeIdPositionPair current_pair;
		s >> n_id >> current_pair.second;
		current_pair.first = NodeId(n_id);
		_vec_id_position[current_pair.first._id_value-1] = current_pair;
	}

	s >> str_footer;
	if ( str_footer != "</NodeIdPosition>")
		throw NeuronIdPositionParsingException(string("NodeIdPosition footer expected"));
}

NodeIdPosition& NodeIdPosition::operator=(const NodeIdPosition& rhs)
{
	if (this == &rhs)
		return *this;

	_vec_desc        = rhs._vec_desc;
	_vec_desc_data   = rhs._vec_desc_data;
	_vec_id_position = rhs._vec_id_position;

	return *this;
}

ForwardOrder NodeIdPosition::begin() const
{
	return ForwardOrder(0,_vec_id_position);
}

ForwardOrder NodeIdPosition::end() const
{
	return ForwardOrder(static_cast<Index>(_vec_id_position.size()),_vec_id_position);
}

RZOrder NodeIdPosition::rzbegin() const
{
	return RZOrder(0,_vec_id_position,_vec_desc);
}

RZOrder NodeIdPosition::rzend() const
{
	return RZOrder(static_cast<Index>(_vec_id_position.size()),_vec_id_position,_vec_desc);
}

void NodeIdPosition::ReverseZPositions()
{
	// First determine the highest z-value
	//
	Index z_max = 0;
	for
	(
		vector<NodeIdPositionPair>::iterator iter_max = _vec_id_position.begin();
		iter_max != _vec_id_position.end();
		iter_max++
	)
		if (iter_max->second._position_z > z_max)
			z_max = iter_max->second._position_z;

	for
	(
		vector<NodeIdPositionPair>::iterator iter = _vec_id_position.begin();
		iter != _vec_id_position.end();
		iter++
	)
		iter->second._position_z = z_max - iter->second._position_z;
}