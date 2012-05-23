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
#ifdef WIN32
#pragma warning(disable:4267)
#pragma warning(disable:4996)
#endif 

#include <cassert>
#include "WeightList.h"

/*
ConversionIdList::ConversionIdList()
{
}

void ConversionIdList::Clear(ClamLib::Number n_nodes)
{
//	if (n_nodes == 0)
//		throw ClamLibException(STR_NOTHING_TO_CONVERT);

	// make space for NodeId(0) as well ...
	// TODO: look at this, this is an error which is easily repeated
	_association = vector<CircuitInfo>(n_nodes + 1);
}

void ConversionIdList::AssociateWithDynamicNodes
(	
	Id id,
	const CircuitInfo& info
)
{
	assert(id._id_value < static_cast<int>(_association.size()));
	_association[id._id_value] = info;
}

const CircuitInfo& ConversionIdList::operator [](Id id) const
{
	return _association[id._id_value];
}
*/
/*
link_vector ConversionIdList::Pair
(
	Id		id_out,
	Id		id_in,
	ClamLib::Efficacy	weight
) const
{
	link_vector list_return;

	CircuitInfo info_out = _association[id_out._id_value];
	CircuitInfo info_in  = _association[id_in._id_value];

	if ( info_out.IsSingleId()  && info_in.IsSingleId() )
	{
		// simple node connection

		Id id_out = info_out.GetSingleId();
		Id id_in  = info_in.GetSingleId();

		list_return.push_back(D_WeightedLink(id_in,id_out,weight));
	}

	if ( ! info_out.IsSingleId() && info_in.IsSingleId() )
	{
		// this can only be a connection from an input node to two circuit nodes
		Id id_in = info_in.GetSingleId();

		// if the weight is positive only a connection to the p-inputs has to be made
		if ( weight >= 0)
		{

			list_return.push_back(D_WeightedLink(id_in, info_out._id_ep, weight));
			list_return.push_back(D_WeightedLink(id_in, info_out._id_ip, weight));
		}
		// else to n-inputs
		else
		{
			// weight is negative, but connections are exc to exc
			list_return.push_back(D_WeightedLink(id_in, info_out._id_en, -weight));
			list_return.push_back(D_WeightedLink(id_in, info_out._id_in, -weight));
		}
	}

	if ( info_out.IsSingleId() && ! info_in.IsSingleId() )
	{
		Id id_out = info_out.GetSingleId();

		if (weight >= 0)
			list_return.push_back(D_WeightedLink(info_in._id_p_out, id_out, weight));
		else
			// see above
			list_return.push_back(D_WeightedLink(info_in._id_n_out, id_out, -weight));		
	}

	if (! info_out.IsSingleId() && !info_in.IsSingleId() )
	{
		if (weight >=0)
		{
			//straight connections: n to n, p to p
			list_return.push_back(D_WeightedLink(info_in._id_p_out, info_out._id_ep,weight));
			list_return.push_back(D_WeightedLink(info_in._id_p_out, info_out._id_ip,weight));
			list_return.push_back(D_WeightedLink(info_in._id_n_out, info_out._id_en,weight));
			list_return.push_back(D_WeightedLink(info_in._id_n_out, info_out._id_in,weight));
		}
		else
		{
			//cross connections: n to p, p to n
			list_return.push_back(D_WeightedLink(info_in._id_p_out, info_out._id_en,-weight));
			list_return.push_back(D_WeightedLink(info_in._id_p_out, info_out._id_in,-weight));
			list_return.push_back(D_WeightedLink(info_in._id_n_out, info_out._id_ep,-weight));
			list_return.push_back(D_WeightedLink(info_in._id_n_out, info_out._id_ip,-weight));
		}
	}

	return list_return;
}
*/