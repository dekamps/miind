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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#include <cassert>
#include <MPILib/include/largeNetwork/LargeNetwork.hpp>
#include <MPILib/include/largeNetwork/GenerateHexagonalNetwork.hpp>

#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/PopulationAlgorithm.hpp>
#include <MPILib/include/populist/RateFunctor.hpp>
using namespace MPILib;
using namespace largeNetwork;
namespace {

Rate CorticalBackground(Time t) {
	return RATE_TWOPOPULATION_EXCITATORY_BACKGROUND;
}
// Provide a sharp burst to centre

Rate Burst(Time t) {
	return (t > 0.08 && t < 0.090) ? 5.0 : 0.0;
}

void Add_J_II(MPILib::populist::Pop_Network* p_net,
		const std::vector<NodeId>& vec) {

	populist::Pop_Network::WeightType connection_J_II(TWOPOPULATION_C_I,
			-TWOPOPULATION_J_II);

	for (Index i = 0; i < vec.size(); i++) {

		p_net->makeFirstInputOfSecond(vec[i], vec[i], connection_J_II);
	}
}

void Add_J_EI(MPILib::populist::Pop_Network* p_net,
		const std::vector<IdGrid>& vec_grid,
		const std::vector<NodeId>& vec_link) {
	assert(vec_grid.size() == vec_link.size());

	// I to E
	populist::Pop_Network::WeightType connection_J_EI(TWOPOPULATION_C_I,
			-TWOPOPULATION_J_EI);

	for (Index i = 0; i < vec_grid.size(); i++) {
		p_net->makeFirstInputOfSecond(vec_link[i], vec_grid[i]._id,
				connection_J_EI);
	}
}

void Add_J_IE(
		MPILib::populist::Pop_Network* p_net,
		const std::vector<IdGrid>& vec_grid,
		const std::vector<NodeId>& vec_link) {
	populist::Pop_Network::WeightType connection_J_IE(
			static_cast<Number>(TWOPOPULATION_C_E * 0.5), // other half should come from cortical background
			TWOPOPULATION_J_IE);

	for (Index i = 0; i < vec_grid.size(); i++)
		p_net->makeFirstInputOfSecond(vec_grid[i]._id, vec_link[i],
				connection_J_IE);

}

void Add_J_IE_bg(MPILib::populist::Pop_Network* p_net, NodeId id_bg,
		const std::vector<NodeId>& vec_link) {

	populist::Pop_Network::WeightType connection_J_IE_BG(
			static_cast<Number>(TWOPOPULATION_C_E * 0.5), TWOPOPULATION_J_IE);

	for (Index i = 0; i < vec_link.size(); i++)
		p_net->makeFirstInputOfSecond(id_bg, vec_link[i], connection_J_IE_BG);

}

void Add_J_EE_bg(MPILib::populist::Pop_Network* p_net, NodeId id_bg,
		const std::vector<IdGrid>& vec_grid) {
	populist::Pop_Network::WeightType connection_J_EE_BG(
			TWOPOPULATION_C_E * (0.5), TWOPOPULATION_J_EE);

	for (Index i = 0; i < vec_grid.size(); i++)
		p_net->makeFirstInputOfSecond(id_bg, vec_grid[i]._id,
				connection_J_EE_BG);
}

void Add_J_EE(MPILib::populist::Pop_Network* p_net, const std::vector<IdGrid>& vec_grid,
		const std::vector<nodepair>& vec_link) {
	// Excitatory connection to itself

	for (Index i = 0; i < vec_grid.size(); i++) {
		Number n_neighbours =
				NodesOntoThisNode(vec_link, vec_grid[i]._id).size();

		populist::Pop_Network::WeightType connection_J_EE(
				TWOPOPULATION_C_E * 0.5 / (n_neighbours + 1),
				TWOPOPULATION_J_EE);

		p_net->makeFirstInputOfSecond(vec_grid[i]._id, vec_grid[i]._id,
				connection_J_EE);
	}
}

void Add_Lateral(MPILib::populist::Pop_Network* p_net, const std::vector<IdGrid>& vec_grid,
		const std::vector<nodepair>& vec_link) {
	for (Index i = 0; i < vec_grid.size(); i++) {
		std::vector<NodeId> vec_neighbour = NodesOntoThisNode(vec_link,
				vec_grid[i]._id);
		Number n_neighbours = vec_neighbour.size();

		for (Index j_in = 0; j_in < n_neighbours; j_in++) {
			populist::Pop_Network::WeightType connection_J_EE(
					TWOPOPULATION_C_E * 0.5 / (n_neighbours + 1),
					TWOPOPULATION_J_EE);
			p_net->makeFirstInputOfSecond(vec_neighbour[j_in], vec_grid[i]._id,
					connection_J_EE);
		}
	}
}
}

void GenerateHexagonalNetwork(Number n_rings,				//! number of rings
		MPILib::populist::Pop_Network* p_net,		//! network to which populations should be added
		NodeId* p_id_cent,				//! id of the central node id
		NodeId* p_id_bg,				//! id of the background node
		std::vector<IdGrid>* p_vec_grid,//! list of Ids and positions for the excitatory nodes in the hexagon
		std::vector<std::pair<NodeId, NodeId> >* p_vec_link,//! list of neighbours for the excitatory nodes
		std::vector<NodeId>* p_vec_inh,				//! list of inhibitory nodes
		int* p_offset		//! offset between excitatory and inhibitory nodes
		) {
	BuildHexagonalGrid(p_vec_grid, p_vec_link, n_rings);

	populist::PopulationAlgorithm alg_e(TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER_POP);

	*p_id_cent = NodeId(1);

	for (Index i = 0; i < p_vec_grid->size(); i++) {
		NodeId id_e = p_net->addNode(alg_e, EXCITATORY);
		assert(id_e == (*p_vec_grid)[i]._id);
	}

	populist::PopulationAlgorithm alg_i(TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER_POP);
	for (Index i = 0; i < p_vec_grid->size(); i++)
		p_vec_inh->push_back(p_net->addNode(alg_i, INHIBITORY));

	*p_offset = (*p_vec_inh)[0] - (*p_id_cent); // one being the id value of the central id node

	// Create cortical background, and add to network
	populist::RateFunctor < populist::Pop_Network::WeightType
			> cortical_background(CorticalBackground);
	*p_id_bg = p_net->addNode(cortical_background, EXCITATORY);

	Add_J_II(p_net, *p_vec_inh);
	Add_J_EI(p_net, *p_vec_grid, *p_vec_inh);
	Add_J_IE(p_net, *p_vec_grid, *p_vec_inh);
	Add_J_IE_bg(p_net, *p_id_bg, *p_vec_inh);
	Add_J_EE_bg(p_net, *p_id_bg, *p_vec_grid);
	Add_J_EE(p_net, *p_vec_grid, *p_vec_link);
	Add_Lateral(p_net, *p_vec_grid, *p_vec_link);

	populist::RateFunctor < populist::Pop_Network::WeightType > burst(Burst);
	NodeId id_burst = p_net->addNode(burst, EXCITATORY);
	populist::Pop_Network::WeightType connection_J_EE_Burst(
			BURST_FACTOR * TWOPOPULATION_C_E, TWOPOPULATION_J_EE);

	p_net->makeFirstInputOfSecond(id_burst, *p_id_cent, connection_J_EE_Burst);
}
