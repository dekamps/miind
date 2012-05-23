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
#ifndef _CODE_LIBS_POPULISTLIB_TESTPOPULIST_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTPOPULIST_INCLUDE_GUARD

//! Author: Marc de Kamps
//! Date: 26.01.04
//! Short description: Central test suite for the Populist library

#include <iostream>
#include "../UtilLib/UtilLib.h"
#include "DiffusionZeroLeakEquations.h"
#include "NumericalZeroLeakEquations.h"
#include "OrnsteinUhlenbeckConnection.h"
#include "OrnsteinUhlenbeckParameter.h"
#include "PopulationAlgorithm.h"
#include "PopulistSpecificParameter.h"
#include "TestOmurtagDefinitions.h"
#include "VMatrixCode.h"

using DynamicLib::D_DynamicNetwork;
using DynamicLib::RootReportHandler;
using DynamicLib::SimulationRunParameter;
using std::ostream;


namespace PopulistLib
{

	//! Test class for the Populist library

	class TestPopulist : public LogStream
	{
	      public:

		/// constructor
		TestPopulist 
		(
			boost::shared_ptr<ostream>	/// requires a stream 
		);

		virtual ~TestPopulist ();

		/// Execute test class, return value indicates success of test suite
		bool Execute ();

		bool GenericOnePopulationTest
			(
				Rate,
				Efficacy,
				const PopulationParameter&,
				const SimulationRunParameter&,
				const PopulistSpecificParameter&,
				bool b_log = false
			) const;
	
		bool GenericTwoPopulationTest
		(
			Rate,
			Efficacy,
			Rate,
			Efficacy,
			const PopulationParameter&,
			const SimulationRunParameter&,
			const PopulistSpecificParameter&,
			bool b_log = false
		) const;

		template <class ZeroLeakEquations>
		bool ResponseCurveSingle
		(
		 bool,
			bool,
			bool,
			Index,
			Potential
		) const;

		template <class ZeroLeakEquations>
		bool ResponseCurveDouble
		(
			bool,
			bool,
			bool,
			Index,
			Potential,
			Number
		) const;

		string ResponseCurveFileName
		(
			bool, //!< true if double input training files are requested
			bool, //!< true if refractive training files are requested
			Index //!< which of the mu points is requested?
		) const;

	private:

		template <class ZeroLeakEquations>
		void ResponseCurveSingleNetwork
		(
			Pop_Network*,
			PopulistSpecificParameter*,
			bool,
			bool,
			bool,
			Rate*,
			NodeId*
		) const;

		template <class ZeroLeakEquations>
		void ResponseCurveDoubleNetwork
		(
			Pop_Network*,
			PopulistSpecificParameter*,
			bool,
			bool,
			bool,
			Rate*,
			Rate*,
			NodeId*
		) const;

		bool GenerateVLookUp			() const;
		bool InnerProductTest			() const;
		bool BinCalculationTest			() const;
		bool ZeroLeakTest				() const;
		bool ZeroLeakFluxTest			() const;
		bool InitialDensityTest			() const;

		bool SingleInputZeroLeakEquationsTest() const;

		bool OldOmurtagetAlTest			() const;
		bool OmurtagetAlTest			() const;
		bool OmurtagRefractiveZeroTest	() const;
		bool OmurtagRefractiveTest		() const;
		bool OmurtagFitTest				() const;
		bool OmurtagPolynomialTest		() const;
		bool OmurtagMatrixTest			() const;
		bool OmurtagNumericalTest		() const;

		bool OmurtagNumericalRefractiveTest	() const;

		bool ZeroLeakBuilderTest		() const;
		bool GammaZTest					() const;
		bool Vkj3Test					() const;
		bool NonCirculantTransferTest	() const;
		bool VArrayTest					() const;
		bool ChebyshevVTest				() const;
		bool InhibitionTest             () const;
		bool ZeroLeakGaussTest          () const;
		bool DoubleRebinnerTest			() const;
		bool OmurtagDoubleTest			() const;
		bool LeakGaussTest				() const;
		bool PotentialToBinTest			() const;
		bool OrnsteinUhlenbeckProcessTest() const;
		bool TwoPopulationTest			() const;
		bool GenerateVDataTest			() const;
		bool ResponseCurveExample		() const;
		bool PrintResponseCurve			() const;
		bool ScalarProductTest			() const;
		bool AEIFIntegratorTest			() const;
		bool OneDMTest					() const;
		bool HazardFunctionTest			() const;
		bool BalancedExample			() const;
		bool InputConvertorTest			() const;
		bool RootFileInterpreterTest	() const;

		bool StreamPopulationAlgorithmOut	() const;
		bool StreamPopulationAlgorithmIn	() const;
		bool StreamOUAlgorithmIn			() const;
		bool StreamOUAlgorithmOut			() const;
		bool BuildPopulationAlgorithm		() const;
		bool BuildRateAlgorithm				() const;
		bool BuildOUAlgorithm				() const;

		bool TestResponseCurveSingle				(bool) const;
		bool TestResponseCurveDouble				(bool) const;

		void ResponseSpecificParameter
		(	
			bool,					
			bool,						
			bool,						
			PopulistSpecificParameter*,
			Number n = 3,
			double diff_lim = 0.05
		) const;

		void ResponseSingleRunParameter
		(
			bool,
			bool,
			bool,
			bool,
			Index,
			string&,
			string&,
			PopulistSpecificParameter*,
			Number
		 ) const;

		void ProcessResults();

	}; // end of TestPopulist


}	// end of PopulistLib namespace

#endif	// include guard
