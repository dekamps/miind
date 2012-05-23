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
#ifndef _CODE_LIBS_CLAMLIB_CLAMLIBTEST_INCLUDE_GUARD
#define _CODE_LIBS_CLAMLIB_CLAMLIBTEST_INCLUDE_GUARD

#include <string>
#include "ClamLib.h"
#include "../DynamicLib/DynamicLib.h"
#include "../UtilLib/UtilLib.h"
#include "../StructnetLib/StructnetLib.h"
#include "AddTrainedNetToDynamicNetwork.h"
#include "RootLayeredNetDescription.h"
#include "SimulationInfoBlock.h"
#include "TrainedNet.h"

using DynamicLib::D_DynamicNetwork;
using UtilLib::LogStream;
using StructnetLib::SpatialConnectionistNet;
using std::string;

namespace ClamLib {

	class ClamLibTest : public LogStream {
	public:

		ClamLibTest
		(
			boost::shared_ptr<ostream>,
			const string& path = ""
		);

		bool Execute();

	private:

		bool SemiSigmoidWriteTest			() const;
		bool SemiSigmoidReadTest			() const;
		bool SemiSigmoidBuildTest			() const;
		bool TrainedNetStreamingTest		() const;
		bool PositiveConversionTest			() const;
		bool NegativeConversionTest			() const;
		bool SmallNetPositiveConversionTest	() const;
		bool SmallNetNegativeConversionTest	() const;
		bool NetworkDevelopmentTest			() const;
		bool TestNetworkDevelopmentTest		() const;
		bool CheckReverseNetTraining		() const;
		bool JOCNFWDNetConversionTest		() const;
		bool JOCNFunctorTest				() const;
		bool JOCNConversionTest				() const;
		bool JOCNCorrespondenceTest			() const;
		bool JOCNDisinhibitionTest			() const;
		bool RootWriteLayerTest				() const;
		bool RootWriteLayerVectorTest		() const;
		bool MetaNetTest					() const;
		bool SmallNetMetaTest				() const; 
		bool WeightLinkWriteTest			() const;
		bool CircuitInfoWriteTest			() const;
		bool CircuitInfoReadTest			() const;
		bool IdWriteTest					() const;
		bool SimulationInfoBlockWriteTest	() const;
		bool SimulationInfoBlockReadTest	() const;
		bool CircuitCreatorProxyTest		() const;
		bool CircuitFactoryTest				() const;
		bool SmallPositiveInfoTest			() const;
		bool ConfigurableCreatorTest		() const;
		bool TestStockDescriptions			() const;

		bool PerceptronConfigurableCreatorTest		() const;
		bool SimulationInfoBlockVectorWriteTest		() const;
		bool SimulationInfoBlockVectorReadTest		() const;
		bool SimulationOrganizerSmallDirectTest		() const;
		bool SimulationInfoJOCNFFDTest				() const;
		bool SimulationInfoJOCNTest					() const;
		bool SimulationInfoJOCNFFDConfigurableTest	() const;
		bool SimulationInfoJOCNConfigurableTest		() const;
		bool DynamicSubLayeredIteratorTest			() const;
		bool DynamicSubLayeredReverseIteratorTest	() const;		
		bool JOCNIteratorTest						() const;

		bool SubNetworkIteratorTest			() const;
		bool ReverseSubNetworkIteratorTest	() const;
		bool SimulationResultIteratorTest	() const;

		string GenerateSmallSimulationFile	() const;
	
		bool IndexWeightSerializationTest	() const;

		bool CircuitNodeRoleSerializationTest	() const;
		bool DisInhibitionTest					() const;

		enum ConversionMode { DIRECT, CIRCUIT };

		bool GenericJocnTest
		(
			const string&,
			D_DynamicNetwork*,
			const AbstractCircuitCreator&,
			SimulationOrganizer* p_org = 0,
			bool ffd = true
		) const;

		bool GenerateConversionTest
		(
			const ConversionMode&,
			const TrainedNet&,
			D_DynamicNetwork*
		 ) const;

		bool OrganizerConversionTest
		(
			const string&,
			const ConversionMode&,
			const TrainedNet&,
			D_DynamicNetwork*,
			auto_ptr<SimulationInfoBlock>&
		 ) const;

		D_DynamicNetwork GenerateJocnDynamicNet
		(
			const D_Pattern& pat_in,
			const D_Pattern& pat_out,
			SimulationOrganizer*
		) const;

		TrainedNet GenerateSmallDirectMetaNet() const;

		enum HandlerMode {ASCII, ROOT };

		bool RunTestSimulationSet
			(
				D_DynamicNetwork*,
				const string&,
				HandlerMode
			) const;

		TrainedNet GenerateTrainedNet(const ConversionMode&) const;
		TrainedNet GenerateSmallNet  (const ConversionMode&) const;


	};
}

#endif // include guard
