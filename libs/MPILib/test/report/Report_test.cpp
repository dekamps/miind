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

//Hack to test privat members
#define private public
#define protected public
#include <MPILib/include/report/Report.hpp>
#undef protected
#undef private
#include <MPILib/include/report/ReportType.hpp>

#include <MPILib/include/algorithm/AlgorithmGrid.hpp>

#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report;


void test_Constructor() {
	Report r(1.0, 2.0, 1);

	BOOST_CHECK(r._id ==1);
	BOOST_CHECK(r._rate==2.0);
	BOOST_CHECK(r._time==1.0);
	ReportValue tempV {"blab", 1.0, 2.0};
	std::vector<ReportValue> rv;
	rv.push_back(tempV);
	Report r1(1.0, 2.0, 1,  MPILib::algorithm::AlgorithmGrid(2), RATE, rv);

	BOOST_CHECK(r1._id==1);
	BOOST_CHECK(r1._grid.getStateSize()==2);
	BOOST_CHECK(r1._rate==2.0);
	BOOST_CHECK(r1._time==1.0);
	BOOST_CHECK(r1._type==RATE);
	BOOST_CHECK(r1._values[0]._name_quantity=="blab");
	BOOST_CHECK(r1._nrNodes == 0);

	Report r2(1.0, 2.0, 1,  MPILib::algorithm::AlgorithmGrid(2), RATE, rv, 10);

	BOOST_CHECK(r2._id==1);
	BOOST_CHECK(r2._grid.getStateSize()==2);
	BOOST_CHECK(r2._rate==2.0);
	BOOST_CHECK(r2._time==1.0);
	BOOST_CHECK(r2._type==RATE);
	BOOST_CHECK(r2._values[0]._name_quantity=="blab");
	BOOST_CHECK(r2._nrNodes == 10);
}

void test_addValue(){
	Report r(1.0, 2.0, 1);
	ReportValue tempV {"blab", 1.0, 2.0};

	r.addValue(tempV);
	BOOST_CHECK(r._values[0]._name_quantity=="blab");

}

int test_main(int argc, char* argv[]) // note the name!
		{

	test_Constructor();
	test_addValue();
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
