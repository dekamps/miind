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
#include "DefaultLayerNodeLinkCollection.h"

using namespace std;
using namespace NetLib;


DefaultLayerNodeLinkCollection::DefaultLayerNodeLinkCollection
(
		const vector<Layer>& vector_of_layers
):
_b_is_valid(true),
_number_of_nodes(accumulate(vector_of_layers.begin(),vector_of_layers.end(),0,plus<Layer>())),
_vec_link(InitializeNodeLinks(vector_of_layers))
{
}


DefaultLayerNodeLinkCollection::DefaultLayerNodeLinkCollection
(
	const DefaultLayerNodeLinkCollection& rhs
):
_b_is_valid(rhs._b_is_valid),
_number_of_nodes(_b_is_valid ? rhs._number_of_nodes : 0),
_vec_link(_b_is_valid? rhs._vec_link : vector<NodeLink>(0))
{
}

NodeLink DefaultLayerNodeLinkCollection::pop()
{
	// Ignore the slight inefficiency by this statement
	_b_is_valid = false;

	NodeLink link_ret = _vec_link[_vec_link.size()-1];
	_vec_link.pop_back();
	return link_ret;
}

DefaultLayerNodeLinkCollection* DefaultLayerNodeLinkCollection::Clone() const
{
	return new DefaultLayerNodeLinkCollection(*this);
}

DefaultLayerNodeLinkCollection::~DefaultLayerNodeLinkCollection()
{
}

Number DefaultLayerNodeLinkCollection::NumberOfPredecessors(NodeId id) const
{
	return 0;
}

Number DefaultLayerNodeLinkCollection::NumberOfNodes() const
{
	return static_cast<Number>(_vec_link.size());
}

bool DefaultLayerNodeLinkCollection::IsValid() const
{
	return _b_is_valid;
}


vector<NodeLink> DefaultLayerNodeLinkCollection::InitializeNodeLinks
(
	const vector<Layer>& vec_layer
)
{
		vector<Index> vec_begin(vec_layer.size() + 1);
		vec_begin[0] = 1;

		// create a vector who starts with node id 1, and then has the first node id of the next highest layers
		// the highest element has  a node id of 1 past the number of nodes and serves to break off the loop
		copy
		(
			vec_layer.begin(),
			vec_layer.end(),
			vec_begin.begin()+1
		);

		partial_sum
		(
			vec_begin.begin(),
			vec_begin.end(),
			vec_begin.begin()
		);

		
		vector<NodeLink> vec_link;

		// start with the first layer, they do not have inputs, unless there is a threshold
		Layer layer = 0;

		vector<NodeId> vec_id(0);


		for (Index i = vec_begin[0]; i < vec_begin[1]; i++ )
		{
			NodeLink link(NodeId(i),vec_id);
			vec_link.push_back(link);

		}

		// now deal with the higher layers
		for (layer = 1; layer < vec_layer.size(); layer++)
		{
			// vec_begin[layer+1] is always defined because
			// the size of vec_begin is one larger than that of vec_layer
			for (Index ind = vec_begin[layer]; ind < vec_begin[layer+1]; ind++) 
			{
				// first get the input nodes
				vec_id.clear();

				// vec_begin[layer-1] is defined because layer is at least one
				for (Index input = vec_begin[layer-1]; input < vec_begin[layer]; input++)
					vec_id.push_back(NodeId(input));
		
				NodeLink link(NodeId(ind),vec_id);
				vec_link.push_back(link);
			}
		}
		return vec_link;
}