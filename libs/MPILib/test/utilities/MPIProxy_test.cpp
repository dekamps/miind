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

#define private public
#define protected public
#include <MPILib/include/utilities/MPIProxy.hpp>
#undef protected
#undef private

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	MPIProxy mpiProxy;

	BOOST_CHECK(mpiProxy._rank == world.rank());
	BOOST_CHECK(mpiProxy._size == world.size());
}

void test_Getter() {
	MPIProxy mpiProxy;
	BOOST_CHECK(mpiProxy.getRank() == world.rank());
	BOOST_CHECK(mpiProxy.getSize() == world.size());

}

void test_Broadcast(){

	int blub = 0;
	BOOST_CHECK(blub == 0);
	if(world.rank()==0){
		blub = 9;
		BOOST_CHECK(blub == 9);
	}
	MPIProxy mpiProxy;
	mpiProxy.broadcast(blub, 0);

	BOOST_CHECK(blub == 9);

	blub = 0;
	BOOST_CHECK(blub == 0);
	if(world.rank()==0){
		blub = 9;
		BOOST_CHECK(blub == 9);
	}
	mpiProxy.broadcast(blub, 1);

	BOOST_CHECK(blub == 0);
}

int test_main(int argc, char* argv[]) // note the name!
		{

	mpi::environment env(argc, argv);

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	// we use only two processors for this testing
	test_Constructor();
	test_Getter();
	test_Broadcast();

	return 0;

}

