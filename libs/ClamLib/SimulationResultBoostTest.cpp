// Copyright (c) 2005 - 2011 Dave Harrison
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
//
#define BOOST_TEST_MODULE SimulationResultTest

#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <TFile.h>
#include <TGraph.h>

#include <valarray>
#include <numeric>

#include "../NumtoolsLib/NumtoolsLib.h"

#include "Id.h"
#include "SimulationResult.h"

using NumtoolsLib::NumtoolsException;
using ClamLib::SimulationResult;

class SimulationResultTest
{
	public:
        SimulationResultTest() :
        	file("test/simulationjocn.root"),
            simresult(file)
        {
        }

        ~SimulationResultTest()
        {
        	file.Close();
        }

        TFile file;
		SimulationResult simresult;

		static const double OVERFLOW_DOUBLE, UNDERFLOW_DOUBLE, TIME, TIME_RESOLUTION, EXPECTED_RESULT;
		static const ClamLib::Id ID, OVERFLOW_ID, UNDERFLOW_ID;
};

const double SimulationResultTest::OVERFLOW_DOUBLE = std::numeric_limits<double>::max();
const double SimulationResultTest::UNDERFLOW_DOUBLE = -SimulationResultTest::OVERFLOW_DOUBLE;
const double SimulationResultTest::TIME = 0.3891;
const double SimulationResultTest::TIME_RESOLUTION = 0.0001;
const double SimulationResultTest::EXPECTED_RESULT = 0.0499584;

const ClamLib::Id SimulationResultTest::ID(7819);
const ClamLib::Id SimulationResultTest::OVERFLOW_ID(100000);
const ClamLib::Id SimulationResultTest::UNDERFLOW_ID(-3);


BOOST_FIXTURE_TEST_SUITE( s, SimulationResultTest )

/*!
 * Test get known result
 */
BOOST_AUTO_TEST_CASE( RateByIdForTimeEqualsKnownTest )
{
	double result = simresult.RateForIdByTime(ID, TIME);
	BOOST_CHECK_CLOSE( EXPECTED_RESULT, result, 0.0001 );
}

/*!
 * Test get known result
 */
BOOST_AUTO_TEST_CASE( RateByIdForTimeEqualsInterpTest )
{
	double result = simresult.RateForIdByTime(ID, TIME-TIME_RESOLUTION/2.0);
	BOOST_CHECK_CLOSE( EXPECTED_RESULT, result, 0.0001 );
}

/*!
 * Get Rate for time before we have data
 *
 * Should throw NumtoolsException
 */
BOOST_AUTO_TEST_CASE( RateByIdForTimeUnderFlowDataTest )
{
	BOOST_CHECK_THROW(simresult.RateForIdByTime(ID, UNDERFLOW_DOUBLE), NumtoolsLib::NumtoolsException);
}

/*!
 * Get Rate for time after we have data
 *
 * Should throw NumtoolsException
 */
BOOST_AUTO_TEST_CASE( RateByIdForTimeOverFlowDataTest )
{
	BOOST_CHECK_THROW(simresult.RateForIdByTime(ID, OVERFLOW_DOUBLE), NumtoolsLib::NumtoolsException);
}

/*!
 * Get Rate for time before we have data
 *
 * Should throw NumtoolsException
 */
BOOST_AUTO_TEST_CASE( RateByIdForTimeUnderFlowIdTest )
{
	BOOST_CHECK_THROW(simresult.RateForIdByTime(UNDERFLOW_ID, TIME), NumtoolsLib::NumtoolsException);
}

/*!
 * Get Rate for time before we have data
 *
 * Should throw NumtoolsException
 */
BOOST_AUTO_TEST_CASE( RateByIdForTimeOverFlowIdTest )
{
	BOOST_CHECK_THROW(simresult.RateForIdByTime(OVERFLOW_ID, TIME), NumtoolsLib::NumtoolsException);
}

BOOST_AUTO_TEST_SUITE_END()

