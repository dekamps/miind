// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#include <MPILib/config.hpp>
#ifdef ENABLE_MPI
#include <boost/mpi/communicator.hpp>
#endif
#include <MPILib/include/utilities/MPIProxy.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/utilities/CircularDistribution.hpp>
#undef protected
#undef private

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

void test_Constructor() {

	CircularDistribution circularD;

}

void test_isLocalNode() {
	CircularDistribution circularD;

	if (MPIProxySingleton::instance().getSize() == 2) {
		if (MPIProxySingleton::instance().getRank() == 0) {
			BOOST_CHECK(circularD.isLocalNode(0)==true);
			BOOST_CHECK(circularD.isLocalNode(1)==false);

		} else if (MPIProxySingleton::instance().getRank() == 1) {
			BOOST_CHECK(circularD.isLocalNode(0)==false);
			BOOST_CHECK(circularD.isLocalNode(1)==true);
		}
	} else {
		BOOST_CHECK(circularD.isLocalNode(0)==true);
		BOOST_CHECK(circularD.isLocalNode(1)==true);

	}
}

void test_getResponsibleProcessor() {
	CircularDistribution circularD;

	if (MPIProxySingleton::instance().getSize() == 2) {
		BOOST_CHECK(circularD.getResponsibleProcessor(1)==1);
		BOOST_CHECK(circularD.getResponsibleProcessor(0)==0);
	} else {
		BOOST_CHECK(circularD.getResponsibleProcessor(1)==0);
		BOOST_CHECK(circularD.getResponsibleProcessor(0)==0);
	}
}

void test_isMaster() {
	CircularDistribution circularD;

	if (MPIProxySingleton::instance().getRank() == 0) {
		BOOST_CHECK(circularD.isMaster()==true);
	} else {
		BOOST_CHECK(circularD.isMaster()==false);
	}
}

int test_main(int argc, char* argv[]) // note the name!
		{

#ifdef ENABLE_MPI
	boost::mpi::environment env(argc, argv);

	// we use only two processors for this testing
	if (MPIProxySingleton::instance().getSize() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
#endif
	test_Constructor();
	test_isLocalNode();
	test_getResponsibleProcessor();
	test_isMaster();

	return 0;
//    // six ways to detect and report the same error:
//    BOOST_CHECK( add( 2,2 ) == 4 );        // #1 continues on error
//    BOOST_CHECK( add( 2,2 ) == 4 );      // #2 throws on error
//    if( add( 2,2 ) != 4 )
//        BOOST_ERROR( "Ouch..." );          // #3 continues on error
//    if( add( 2,2 ) != 4 )
//        BOOST_FAIL( "Ouch..." );           // #4 throws on error
//    if( add( 2,2 ) != 4 ) throw "Oops..."; // #5 throws on error
//
//    return add( 2, 2 ) == 4 ? 0 : 1;       // #6 returns error code
}
