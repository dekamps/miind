// Copyright (c) 2005 - 2007 Marc de Kamps
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
#include <algorithm>
#include <functional>
#include <cassert>
#include "LayeredArchitecture.h"
#include "DefaultLayerNodeLinkCollection.h"

using namespace NetLib;
using namespace std;

LayeredArchitecture::LayeredArchitecture():
AbstractArchitecture(0),
_b_external_collection(false),
_number_of_nodes(0),
_vector_of_layers(0),
_vector_of_begin_ids(0),
_vector_number_from(0)
{
}

LayeredArchitecture::LayeredArchitecture
(
	const vector<Layer>& vector_of_layers,
	bool b_threshold
):
AbstractArchitecture
(
	new DefaultLayerNodeLinkCollection	//LayerArchitecture owns, because
	(									// for Architecture, this is an external collection
		vector_of_layers
	),
	b_threshold
),
_b_external_collection(false),
_number_of_nodes(NumberInLayers(vector_of_layers)),
_vector_of_layers(vector_of_layers),
_vector_of_begin_ids(InitializeBeginIds(vector_of_layers)),
_vector_number_from(InitializeNumberFrom())
{
}

LayeredArchitecture::LayeredArchitecture
(
	const vector<Layer>& vector_of_layers,
	AbstractNodeLinkCollection* p_collection,
	bool b_threshold
):
AbstractArchitecture(p_collection,b_threshold),
_b_external_collection(true),
_number_of_nodes(NumberInLayers(vector_of_layers)),
_vector_of_layers(vector_of_layers),
_vector_of_begin_ids(InitializeBeginIds(vector_of_layers)),
_vector_number_from(InitializeNumberFrom())
{
}


LayeredArchitecture::LayeredArchitecture
(
        const LayeredArchitecture& rhs
):
AbstractArchitecture(rhs),
_b_external_collection(rhs._b_external_collection),
_number_of_nodes(rhs._number_of_nodes),
_vector_of_layers(rhs._vector_of_layers),
_vector_of_begin_ids(rhs._vector_of_begin_ids),
_vector_number_from(rhs._vector_number_from)
{
}


LayeredArchitecture::~LayeredArchitecture()
{
}

Number LayeredArchitecture::NumberOfLayers() const
{
	return static_cast<Number>(_vector_of_layers.size());
}

Number LayeredArchitecture::NumberOfInputNodes() const
{
	return _vector_of_layers[0];
}

Number LayeredArchitecture::NumberOfOutputNodes() const
{
	return _vector_of_layers[_vector_of_layers.size()-1];
}

vector<Layer> LayeredArchitecture::LayerVector() const
{
	return _vector_of_layers;
}

vector<NodeId> LayeredArchitecture::InitializeBeginIds
(
	const vector<Layer>& vector_of_layers
) const
{
	vector<Layer> vector_of_sums = vector_of_layers;
	vector_of_sums.insert(vector_of_sums.begin(),1);

	vector<Layer>::const_iterator 
		iter_end = partial_sum
				(
					vector_of_sums.begin(),
					vector_of_sums.end(),
					vector_of_sums.begin()
				);

	vector<NodeId> vector_of_return(vector_of_sums.size());

	transform
		(
			vector_of_sums.begin(),
			vector_of_sums.end(),
			vector_of_return.begin(),
			ConvertToNodeId
		);


	vector_of_return.erase(--vector_of_return.end());

	return vector_of_return;
}

vector<NodeId> LayeredArchitecture::BeginIdVector() const
{
	assert(_vector_of_begin_ids.size() == NumberOfLayers());
	return _vector_of_begin_ids;
}

Number LayeredArchitecture::NumberInLayers(const vector<Layer>& vector_of_layers) const
{
	return accumulate
		(
			vector_of_layers.begin(),
			vector_of_layers.end(),
			0,
			plus<Layer>()
		);
}

vector<Number> LayeredArchitecture::InitializeNumberFrom() const
{
	vector<Number> vector_return(NumberOfLayers()-1);

	Layer layer = 0;
	for
	(
		vector<Number>::iterator iter = vector_return.begin();
		iter != vector_return.end();
		iter++, layer++
	)
	{
		Layer next_highest_layer = layer + 1;
		*iter = ConnectionCount
				(
					_vector_of_begin_ids[next_highest_layer],
					_vector_of_layers[next_highest_layer]
				);
	}

	return vector_return;
}

vector<Number> LayeredArchitecture::NumberConnectionsVector() const
{
	return _vector_number_from;
}
