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
#pragma warning(disable: 4267)
#pragma warning(disable: 4996)
#endif

#include <algorithm>
#include "../StructnetLib/StructnetLib.h"
#include <TObjArray.h>
#include "AddTrainedNetToDynamicNetwork.h"
#include "ClamLibException.h"
#include "NetId.h"
#include "SimulationOrganizer.h"
#include "SimulationInfoBlockVector.h"
#include "ToRootLayeredNetDescription.h"

#include <TFile.h>
using namespace ClamLib;
using namespace StructnetLib;
using namespace std;

SimulationOrganizer::SimulationOrganizer():
_root_file_name(""),
_p_dnet(0),
_vec_sub(0)

{
}

NetId SimulationOrganizer::Convert
(
			const string&					name,		//!< Name of the subnetwork
			const TrainedNet&				tn,			//!< Static net (TrainedNet) to be converted into a DynamicNet
			const AbstractCircuitCreator&	creator,
			const Pattern<Rate>&			pat_rate,	//!< Input pattern, specifying the rates that will be offered to the input layer from t = 0 onwards
			D_DynamicNetwork*				p_dnet,		//!< DynamicNetwork in which the static one will be converted
			RateFunction					func_rate,	//!< If this arguments is present, all non-zero elements of the input pattern will be generated according to ths function
			const SpatialPosition&			pos			//!< The SpatialPosition of every node in the DynamicNetwork will be offset by this value
)
{

	// do the actual conversion
	ClamLib::AddTNToDN converter;

	converter.Convert
	(
			tn,			//!< Static net (TrainedNet) to be converted into a DynamicNet
			pat_rate,	//!< Input pattern, specifying the rates that will be offered to the input layer from t = 0 onwards
			creator,
			p_dnet,		//!< DynamicNetwork in which the static one will be converted
			func_rate,	//!< If this arguments is present, all non-zero elements of the input pattern will be generated according to ths function
			pos         //!< The SpatialPosition of every node in the DynamicNetwork will be offset by this value
	);
	// store a reference to the dynamic network
	_p_dnet = p_dnet;

	// extract the LayerDesciption from and add it to the info block
	vector<LayerDescription> vec_desc = tn._net.Dimensions();
	vector<CircuitInfo>      vec_info = converter.CircuitInfoVector();
	SimulationInfoBlock 
		info
		(
			name.c_str(),
			ToRootLayeredNetDescription(vec_desc),
			vec_info,
			creator.Description()
		);
	_vec_sub.push_back(info);

	return NetId(_vec_sub.size() ? _vec_sub.size() - 1 : -1 );
}

const DynamicSubNetwork& SimulationOrganizer::operator[](UtilLib::Index ind) const
{
	assert(ind < InfoSize());

	return _vec_sub[ind];
}

Number SimulationOrganizer::InfoSize() const
{
	return static_cast<Number>(_vec_sub.size());
}

string SimulationOrganizer::Name(Index i) const
{
	return string(_vec_sub[i].GetName());
}

bool SimulationOrganizer::Configure
(
	const SimulationRunParameter& par_run
)
{	
	_root_file_name = par_run.Handler().MediumName();
	return _p_dnet->ConfigureSimulation(par_run);
}

bool SimulationOrganizer::Evolve()
{
	WriteSimulationInfoBlock();
	return _p_dnet->Evolve();
}

void SimulationOrganizer::WriteSimulationInfoBlock()
{
	vector<SimulationInfoBlock> vec_siblock;
	copy(_vec_sub.begin(),_vec_sub.end(),back_inserter(vec_siblock));
	SimulationInfoBlockVector vec_block(vec_siblock);
	vec_block.SetName("simulationinfoblockcollection");
	vec_block.Write();
}