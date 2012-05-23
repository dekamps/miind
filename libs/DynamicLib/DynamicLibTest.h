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
#ifndef _CODE_LIBS_DYNAMICLIB_DYNAMICLIBTEST_INCLUDE_GUARD
#define _CODE_LIBS_DYNAMICLIB_DYNAMICLIBTEST_INCLUDE_GUARD

#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "AfferentCode.h"
#include "AsciiReportHandler.h"
#include "DynamicNodeCode.h"
#include "DynamicNetworkCode.h"
#include "DynamicNetworkImplementationCode.h"
#include "RateAlgorithmCode.h"
#include "RateFunctorCode.h"
#include "ReportManagerCode.h"
#include "RootReportHandler.h"
#include "SpatialDynamicNetwork.h"
#include "WilsonCowanAlgorithm.h"

using std::string;
using std::ostream;
using UtilLib::LogStream;

//! DynamicLib

namespace DynamicLib
{
	//! Central test class for DynamicLib

	class DynamicLibTest : public LogStream
	{
	public:

		//! Constructor using designated stream for logging
		DynamicLibTest
		(
			boost::shared_ptr<ostream>		//! log stream
		);

		//! virtual destructor, required for class deriving from Streamable
		virtual ~DynamicLibTest();

		//! Carry out tests
		bool Execute();

	private:

		D_DynamicNetwork CreateWilsonCowanNetwork() const;
		
		bool GridAndStateStreamingTest	() const;
		bool WilsonCowanAlgorithmTest	() const;
		bool WilsonCowanNetworkTest		() const;
		bool SimpleNetworkTest			() const;
		bool InnerProductTest			() const;
		bool NetworkStreamingTest		() const;
		bool RootHandlerTest			() const;
		bool RootHighThroughputHandlerTest	() const;
		bool MultipleRootTest			() const;
		bool MultipleHighThroughputTest	() const;
		bool NetworkCopyTest            () const;
		bool MaxNumberIterationsTest	() const;
		bool SpatialNetworkTest			() const;
		bool WilsonCowanTest			() const;
		bool SimRunParSerializeTest		() const;
		bool RateAlgorithmSerializeTest	() const;
		bool BuildRateAlgorithm			() const;
		bool WilsonCowanSerializeTest	() const;
		bool BuildWilsonCowanAlgorithm	() const;
		bool WilsonCowanExampleTest		() const;
		bool CanvasParameterWriteTest	() const;

		void ProcessResults();

		const string _pathname;

	}; // end of DynamicLibTest

} // end of DynamicLib

#endif // include guard
