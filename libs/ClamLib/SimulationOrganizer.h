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
#ifndef _CODE_LIBS_CLAMLIB_SIMULATIONORGANIZER_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_SIMULATIONORGANIZER_INCLUDE_GUARD

#include <vector>
#include "AbstractCircuitCreator.h"
#include "BasicDefinitions.h"
#include "DynamicSubNetwork.h"
#include "NetId.h"
#include "SimulationInfoBlock.h"
#include "TrainedNet.h"


using DynamicLib::D_AbstractAlgorithm;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::D_RateAlgorithm;
using DynamicLib::Rate;
using DynamicLib::RateFunction;
using DynamicLib::SimulationRunParameter;
using UtilLib::Index;
using std::vector;

namespace ClamLib {

	class SimulationOrganizer {
	public:

		SimulationOrganizer();

		NetId Convert
		(
			const string&,						//!< Name of the converted network
			const TrainedNet&,					//!< Static net (TrainedNet) to be converted into a DynamicNet
			const AbstractCircuitCreator&,		//!< Creator object for this particular conversion
			const Pattern<Rate>&,				//!< Input pattern, specifying the rates that will be offered to the input layer from t = 0 onwards
			D_DynamicNetwork*,					//!< DynamicNetwork in which the static one will be converted
			RateFunction = 0,					//!< If this arguments is present, all non-zero elements of the input pattern will be generated according to ths function
			const SpatialPosition& = NO_OFFSET	//!< The SpatialPosition of every node in the DynamicNetwork will be offset by this value
		);

		const DynamicSubNetwork& operator[](Index) const;

		string Name(Index) const;

		Number InfoSize() const;

		bool Configure(const SimulationRunParameter&);

		bool Evolve();

	private:

		void WriteSimulationInfoBlock();

		string						_root_file_name;
		D_DynamicNetwork*			_p_dnet;
		vector<DynamicSubNetwork>	_vec_sub;
	};
}

#endif