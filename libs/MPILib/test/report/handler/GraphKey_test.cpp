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

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/report/handler/GraphKey.hpp>
#undef protected
#undef private
#include <string>
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report::handler;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_enum(){
	BOOST_CHECK(RATEGRAPH==1);
	BOOST_CHECK(STATEGRAPH==0);
}

void test_Constructor() {
	GraphKey gk;

	BOOST_CHECK(gk._id==0);
	BOOST_CHECK(gk._time==0.0);
	BOOST_CHECK(gk._type==STATEGRAPH);

	GraphKey gk2(1, 2.0);
	BOOST_CHECK(gk2._id==1);
	BOOST_CHECK(gk2._time==2.0);
	BOOST_CHECK(gk2._type==STATEGRAPH);

	std::string key = "grid_3_3.0";
	GraphKey gk3(key);
	BOOST_CHECK(gk3._id==3);
	BOOST_CHECK(gk3._time==3.0);
	BOOST_CHECK(gk3._type==STATEGRAPH);

	std::string key1 = "rate_4";
	GraphKey gk4(key1);
	BOOST_CHECK(gk4._id==4);
	BOOST_CHECK(gk4._time==0.0);
	BOOST_CHECK(gk4._type==RATEGRAPH);

}

void test_generateName() {
	std::string key = "rate_4";
	GraphKey gk(key);

	BOOST_CHECK(gk.generateName()=="rate_4");

	std::string key1 = "grid_3_3.0";
	GraphKey gk1(key1);
	BOOST_CHECK(gk1.generateName()=="grid_3_3");
}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}
	test_enum();
	test_Constructor();
	test_generateName();
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
