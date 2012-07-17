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
#include <MPILib/include/algorithm/AlgorithmGrid.hpp>
#undef protected
#undef private

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::algorithm;

namespace mpi = boost::mpi;

mpi::communicator world;

void test_Constructor() {

	AlgorithmGrid a(100);
	BOOST_CHECK(a._numberState==100);
	BOOST_CHECK(a._arrayState[0]==0.0);
	BOOST_CHECK(a._arrayState[99]==0.0);
	BOOST_CHECK(a._arrayInterpretation[0]==0.0);
	BOOST_CHECK(a._arrayInterpretation[99]==0.0);

	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	AlgorithmGrid a1(dv);
	BOOST_CHECK(a1._numberState==3);
	BOOST_CHECK(a1._arrayState[0]==1.0);
	BOOST_CHECK(a1._arrayState[1]==2.1);
	BOOST_CHECK(a1._arrayState[2]==3.0);

	BOOST_CHECK(a1._arrayInterpretation[0]==0.0);
	BOOST_CHECK(a1._arrayInterpretation[1]==0.0);
	BOOST_CHECK(a1._arrayInterpretation[2]==0.0);

	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a2(dv, dv1);
	BOOST_CHECK(a2._numberState==3);
	BOOST_CHECK(a2._arrayState[0]==1.0);
	BOOST_CHECK(a2._arrayState[1]==2.1);
	BOOST_CHECK(a2._arrayState[2]==3.0);

	BOOST_CHECK(a2._arrayInterpretation[0]==1.2);
	BOOST_CHECK(a2._arrayInterpretation[1]==2.3);
	BOOST_CHECK(a2._arrayInterpretation[2]==3.3);

}

void test_assignment() {
	AlgorithmGrid a(100);
	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	AlgorithmGrid a1(dv);

	a = a1;
	BOOST_CHECK(a._numberState==3);
	BOOST_CHECK(a._arrayState[0]==1.0);
	BOOST_CHECK(a._arrayState[1]==2.1);
	BOOST_CHECK(a._arrayState[2]==3.0);

	BOOST_CHECK(a._arrayInterpretation[0]==0.0);
	BOOST_CHECK(a._arrayInterpretation[1]==0.0);
	BOOST_CHECK(a._arrayInterpretation[2]==0.0);

}

void test_toStateVector() {
	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv, dv1);

	std::vector<double> v;

	v = a.toStateVector();
	BOOST_CHECK(v[0]==1.0);
	BOOST_CHECK(v[1]==2.1);
	BOOST_CHECK(v[2]==3.0);

}

void test_toInterpretationVector() {
	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv, dv1);

	std::vector<double> v;

	v = a.toInterpretationVector();
	BOOST_CHECK(v[0]==1.2);
	BOOST_CHECK(v[1]==2.3);
	BOOST_CHECK(v[2]==3.3);
}

void test_toValarray() {

	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv1);
	std::valarray<double> d;
	d = a.toValarray<double>(dv1);
	BOOST_CHECK(d[0]==1.2);
	BOOST_CHECK(d[1]==2.3);
	BOOST_CHECK(d[2]==3.3);
}

void test_toVector() {
	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv1);
	std::valarray<double> d;
	d.resize(3,0.0);
	d[0]=1.1;
	d[1]=2.2;
	d[2]=3.3;

	dv1 = a.toVector<double>(d, 3);
	BOOST_CHECK(dv1[0]==1.1);
	BOOST_CHECK(dv1[1]==2.2);
	BOOST_CHECK(dv1[2]==3.3);
}

void test_getters() {
	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv, dv1);
	BOOST_CHECK(a.getStateSize()==3);
	BOOST_CHECK(a.getArrayInterpretation()[0]==1.2);
	BOOST_CHECK(a.getArrayState()[1]==2.1);
}

void test_resize() {
	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv, dv1);
	BOOST_CHECK(a.getArrayInterpretation().size()==3);
	BOOST_CHECK(a.getArrayState().size()==3);
	a.resize(6);
	BOOST_CHECK(a.getArrayInterpretation().size()==6);
	BOOST_CHECK(a.getArrayState().size()==6);
}

void test_iterators() {
	std::vector<double> dv;
	dv.push_back(1.0);
	dv.push_back(2.1);
	dv.push_back(3.0);
	std::vector<double> dv1;
	dv1.push_back(1.2);
	dv1.push_back(2.3);
	dv1.push_back(3.3);
	AlgorithmGrid a(dv, dv1);
	BOOST_CHECK((*a.begin_state())==1.0);
	//TODO implement test for the end state
//	BOOST_CHECK((*a.end_state())==0.0);
	BOOST_CHECK((*a.begin_interpretation())==1.2);
	//TODO implement test for the end state
//	BOOST_CHECK((*a.end_interpretation())==0.0);


}

int test_main(int argc, char* argv[]) // note the name!
		{

	boost::mpi::environment env(argc, argv);
	// we use only two processors for this testing

	if (world.size() != 2) {
		BOOST_FAIL( "Run the test with two processes!");
	}

	test_Constructor();
	test_assignment();
	test_toStateVector();
	test_toInterpretationVector();
	test_toValarray();
	test_getters();
	test_resize();
	test_iterators();

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
