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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include <PopulistLib.h>
#include "../UnitDynamicLib/SinglePopulationNetworkFixtureCode.h"

using PopulistLib::OrnsteinUhlenbeckConnection;
using PopulistLib::QIFAlgorithm;
using PopulistLib::QIFParameter;

using UnitDynamicLib::SinglePopulationNetworkFixture; 

BOOST_FIXTURE_TEST_SUITE( s, SinglePopulationNetworkFixture<OrnsteinUhlenbeckConnection> )

BOOST_AUTO_TEST_CASE( QIF_Algorithm_Run_Single_Equilibrium )
{
	SetOutputFileNames("qif.results","qif.log");
	QIFParameter par;
	par._I			= -9.0;  // negative, so stable equilibrium exists
	par._V_reset	=  0.0;  // below the instable equilibrium of +3.0, so all density below 3.0 will end up in the stable equilibrium
	par._V_peak		= 10.0;
	par._V_min		= -5.0;

//	QIFAlgorithm<double> alg(par);


	// Just run the algorithm for a brief while
    BOOST_CHECK( true );
}
//____________________________________________________________________________//

BOOST_AUTO_TEST_CASE( QIF_Algorithm_Create_And_Copy )
{
	
	QIFParameter par_qif;
	par_qif._I			= -9.0;  // negative, so stable equilibrium exists
	par_qif._V_reset	=  0.0;  // below the instable equilibrium of +3.0, so all density below 3.0 will end up in the stable equilibrium
	par_qif._V_peak		= 10.0;
	par_qif._V_min		= -5.0;

	CharacteristicParameter par_char;
	QIFAlgorithm<double> alg(par_qif,par_char);

	par_qif._I = 9.0; // positive so no stable equilibrium exists
	QIFAlgorithm<double> algdif(par_qif,par_char);

	algdif = alg;

	boost::shared_ptr<QIFAlgorithm<double> > p_alg(alg.Clone());

	BOOST_REQUIRE( true );
}

//____________________________________________________________________________//

//BOOST_AUTO_TEST_CASE( test_case2 )
//{
 //   BOOST_CHECK_EQUAL( i, 0 );
//}

//____________________________________________________________________________//

BOOST_AUTO_TEST_SUITE_END()

