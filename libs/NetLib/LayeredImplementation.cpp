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

#include <numeric>
#include <cassert>
#include "LayeredImplementation.h"
#include "BasicDefinitions.h"
#include "LocalDefinitions.h"
#include "NetworkParsingException.h"

using namespace std;
using namespace NetLib;


LayeredImplementation::LayeredImplementation
(
	const LayeredArchitecture& architecture
):
_number_of_connections	(0),
_number_of_nodes		(architecture.NumberOfNodes ()  ),
_vector_architecture	(architecture.LayerVector   ()	),
_vector_begin_connection(architecture.NumberOfLayers()-1),
_vector_begin_id		(architecture.BeginIdVector ()  ),
_vector_originating		(architecture.NumberConnectionsVector())
{



	_number_of_connections = 
		std::accumulate
		(
			_vector_originating.begin(),
			_vector_originating.end(),
			0
		);

	// end with the NumberOfNodes + 1, so as to conveniently calculate the EndId's
	_vector_begin_id.push_back( NodeId(_number_of_nodes+1) );
}

LayeredImplementation::LayeredImplementation(istream& s):
_number_of_nodes(ParseNumberOfNodes(RemoveHeader(s))),
_vector_architecture(ParseArchitectureVector(s)),
_vector_begin_id(ParseBeginIds(s)),
_vector_originating(ParseOriginatingVector(s))
{
	RemoveFooter(s);
}

bool LayeredImplementation::FromStream(istream& s)
{
	_number_of_nodes     = ParseNumberOfNodes(RemoveHeader(s));
	_vector_architecture = ParseArchitectureVector(s);
	_vector_begin_id     = ParseBeginIds(s);
	_vector_originating  = ParseOriginatingVector(s);

	RemoveFooter(s);

	return true;
}

LayeredImplementation::LayeredImplementation(const LayeredImplementation& rhs):
_number_of_connections  (rhs._number_of_connections),
_number_of_nodes        (rhs._number_of_nodes),
_vector_architecture    (rhs._vector_architecture),
_vector_begin_connection(rhs._vector_begin_connection),
_vector_begin_id        (rhs._vector_begin_id),
_vector_originating     (rhs._vector_originating)
{
}

LayeredImplementation::~LayeredImplementation()
{
}

LayeredImplementation&
	LayeredImplementation::operator=(const LayeredImplementation& rhs)
{
	if ( this == &rhs )
		return *this;

	_vector_architecture	= rhs._vector_architecture;
	_vector_originating  	= rhs._vector_originating;
	_vector_begin_id		= rhs._vector_begin_id;
	_vector_begin_connection= rhs._vector_begin_connection;
	_number_of_connections	= rhs._number_of_connections;
	_number_of_nodes        = rhs._number_of_nodes;

	return *this;

}

Number LayeredImplementation::NumberOfLayers() const
{
	return static_cast<Number>(_vector_architecture.size());
}

Number LayeredImplementation::NrConnectionFrom( size_t nr_starting_layer ) const
{
	assert ( nr_starting_layer < NumberOfLayers() - 1);
	return _vector_originating[nr_starting_layer];
}

Number LayeredImplementation::MaxNumberOfNodesInLayer() const
{
	// Tested:  13-10-1999 
	// STL use: 22-07-2004

	vector<Number>::const_iterator iter = 
		max_element
		(
			_vector_architecture.begin(),
			_vector_architecture.end()
		);

	return *iter;
}

NodeId LayeredImplementation::BeginId(Layer n_layer ) const
{
	assert ( n_layer < NumberOfLayers() );
	return _vector_begin_id[n_layer];
}

NodeId LayeredImplementation::EndId(Layer n_layer ) const
{
	assert ( n_layer < NumberOfLayers() );

	// This method relies on the BeginId's beging calculated past 
	// the last layer:
	assert (_vector_begin_id.size() == NumberOfLayers() + 1);

	return NodeId(_vector_begin_id[n_layer + 1]._id_value - 1);
}

Number LayeredImplementation::NumberOfNodesInLayer(Layer n_layer) const
{
	assert (n_layer < NumberOfLayers() );
	return _vector_architecture[n_layer];
}

Number LayeredImplementation::NrConnections() const
{
	return _number_of_connections;
}

vector<Number> LayeredImplementation::ArchVec() const
{
	return _vector_architecture;
}

istream& LayeredImplementation::RemoveHeader(istream& s) const
{
	string str_header;
	s >> str_header;

	if (str_header != LayeredImplementation::Tag())
		throw NetworkParsingException(string("File header not found"));

	return s;
}

istream& LayeredImplementation::RemoveFooter(istream& s) const
{
	string str_footer;
	s >> str_footer;

	if (str_footer != LayeredImplementation::ToEndTag(Tag()))
		throw NetworkParsingException(string("LayeredImplementation file footer expected"));

	return s;
}

bool LayeredImplementation::ToStream(ostream& s) const
{

	s << LayeredImplementation::Tag()<< "\n";
	s << NumberOfTotalNodes() << "\t" << NumberOfLayers() << "\n";


	// Write architecture vector

	s << _tag_architecture << " ";
	for ( Index n_layer_index = 0; n_layer_index < NumberOfLayers(); n_layer_index++ )
		s << NumberOfNodesInLayer(n_layer_index) << "\t";
	s << ToEndTag(_tag_architecture) << "\n";

	// Write the BeginId's of each layer

	s << _tag_beginid << " ";
	for (Index n_tag_index = 0; n_tag_index < NumberOfLayers() + 1; n_tag_index++ )
		s << _vector_begin_id[n_tag_index]._id_value << "\t";
	s << ToEndTag(_tag_beginid) <<"\n";

	// Write number of connections orginating from layer

	s << _tag_originating << " ";
	size_t nr_originating = _vector_originating.size();
	for (Index n_originating_index = 0; n_originating_index < nr_originating; n_originating_index++ )
		s << _vector_originating[n_originating_index] << "\t";
	s << ToEndTag(_tag_originating) << "\n";;


	// Write the file footer
	s << LayeredImplementation::ToEndTag(Tag()) <<"\n";

	return true;

}

Number LayeredImplementation::NumberOfTotalNodes() const
{
	return _number_of_nodes;
}

string LayeredImplementation::Tag() const
{
	return STR_LAYEREDIMPLEMENTATION_TAG;
}

int LayeredImplementation::ParseNumberOfNodes(istream& s) const
{
	// Auxilliary parsing function, for initialization from input streams

	int number_of_nodes;
	s >> number_of_nodes;
	return number_of_nodes;
}

vector<Number> LayeredImplementation::ParseArchitectureVector(istream& s) const
{
	int number_of_layers;
	s >> number_of_layers;

	// Absorb begin tag
	string tag_architecture;
	s >> tag_architecture;
	if ( tag_architecture != _tag_architecture)
		throw NetworkParsingException(string("Architecture tag expected"));

	vector<Number> vector_return(number_of_layers);
	for (int index_layer = 0; index_layer < number_of_layers; index_layer++)
		s >> vector_return[index_layer];
	

	// Absorb end tag
	s >> tag_architecture;
	if ( tag_architecture != ToEndTag(_tag_architecture) )
		throw NetworkParsingException("Architecture tag expected");

	return vector_return;
}

vector<Number> LayeredImplementation::ParseOriginatingVector(istream& s) const
{
	assert (! _vector_architecture.empty());
	Number number_of_input_layers = static_cast<Number>(_vector_architecture.size()) - 1;

	// Absorb begin tag
	string tag_originating;
	s >> tag_originating;
	if ( tag_originating != _tag_originating)
		throw NetworkParsingException(string("Tag originating expected"));

	vector<Number> vector_return(number_of_input_layers);
	for (Index index_layer = 0; index_layer < number_of_input_layers; index_layer++)
		s >> vector_return[index_layer];
	
	// Absorb end tag
	s >> tag_originating;
	if ( tag_originating != ToEndTag(_tag_originating) )
		throw NetworkParsingException(string("Tag originating expected"));

	return vector_return;
}

vector<NodeId> LayeredImplementation::ParseBeginIds(istream& s) const
{
	Number number_of_input_layers = static_cast<Number>(_vector_architecture.size());

	// Absorb begin tag
	string tag_beginid;
	s >> tag_beginid;
	if ( tag_beginid!= _tag_beginid)
		throw NetworkParsingException("tag begin id expected");

	int id_number;
	vector<NodeId> vector_return(number_of_input_layers+1);
	for (Index index_layer = 0; index_layer < number_of_input_layers + 1; index_layer++)
	{
		s >> id_number;
		vector_return[index_layer] = NodeId(id_number);
	}

	// Absorb end tag
	s >> tag_beginid;
	if ( tag_beginid != ToEndTag(_tag_beginid) )
		throw NetworkParsingException("tag beginid expected");

	return vector_return;
}

ostream& NetLib::operator<<(ostream& s, const LayeredImplementation& implementation)
{
	implementation.ToStream(s);
	return s;
}


string NetLib::LayeredImplementation::_tag_architecture("<Architecture>");
string NetLib::LayeredImplementation::_tag_originating ("<NumberConnectionsOriginatingFrom>");
string NetLib::LayeredImplementation::_tag_beginid     ("<BeginIds>");
