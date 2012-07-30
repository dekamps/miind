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
#include <MPILib/include/report/handler/ValueHandlerHandler.hpp>
#undef protected
#undef private
#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::report::handler;


void test_Constructor() {
	ValueHandlerHandler vh;
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==true);
	BOOST_CHECK(vh._vec_quantity.empty()==true);
	BOOST_CHECK(vh._vec_time.empty()==true);

}

void test_addReport() {
	ValueHandlerHandler vh;

	MPILib::report::Report r(1.0, 2.0, 1);
	MPILib::report::ReportValue rv {"blab", 3.0, 4.0};
	r.addValue(rv);
	vh.addReport(r);

	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==false);
	BOOST_CHECK(vh._vec_quantity.empty()==false);
	BOOST_CHECK(vh._vec_time.empty()==false);
	BOOST_CHECK(vh._vec_names[0]=="blab-1");
	BOOST_CHECK(vh._vec_time[0][0]==4.0);
	BOOST_CHECK(vh._vec_quantity[0][0]==3.0);
}

void test_write() {
	ValueHandlerHandler vh;

	MPILib::report::Report r(1.0, 2.0, 1);
	MPILib::report::ReportValue rv {"blab", 3.0, 4.0};
	r.addValue(rv);
	vh.addReport(r);
	vh.write();
	BOOST_CHECK(vh._is_written==true);
	/// @todo DS improve this test

}

void test_Event() {
	ValueHandlerHandler::Event ev { "blub", 1.0, 2.0 };

	ValueHandlerHandler::Event e(ev);
	BOOST_CHECK(e._str=="blub");
	BOOST_CHECK(e._time==1.0);
	BOOST_CHECK(e._value==2.0);

}

void test_isWritten() {
	ValueHandlerHandler vh;
	BOOST_CHECK(vh.isWritten()==false);
	vh._is_written = true;
	BOOST_CHECK(vh.isWritten()==true);
}

void test_reset() {
	ValueHandlerHandler vh;
	ValueHandlerHandler::Event ev { "blub", 1.0, 2.0 };
	vh.distributeEvent(ev);
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==false);
	BOOST_CHECK(vh._vec_quantity.empty()==false);
	BOOST_CHECK(vh._vec_time.empty()==false);
	vh.reset();
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names.empty()==true);
	BOOST_CHECK(vh._vec_quantity.empty()==true);
	BOOST_CHECK(vh._vec_time.empty()==true);
}

void test_distributeEvent() {
	ValueHandlerHandler vh;
	ValueHandlerHandler::Event ev1 { "blub", 1.0, 2.0 };
	vh.distributeEvent(ev1);
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names[0]=="blub");
	BOOST_CHECK(vh._vec_time[0][0]==1.0);
	BOOST_CHECK(vh._vec_quantity[0][0]==2.0);
	ValueHandlerHandler::Event ev { "blub", 3.0, 4.0 };
	vh.distributeEvent(ev);
	BOOST_CHECK(vh._is_written==false);
	BOOST_CHECK(vh._vec_names[0]=="blub");
	BOOST_CHECK(vh._vec_time[0][0]==1.0);
	BOOST_CHECK(vh._vec_quantity[0][0]==2.0);
	BOOST_CHECK(vh._vec_time[0][1]==3.0);
	BOOST_CHECK(vh._vec_quantity[0][1]==4.0);

}

int test_main(int argc, char* argv[]) // note the name!
		{
	test_Constructor();
	test_addReport();
	test_write();
	test_Event();
	test_isWritten();
	test_reset();
	test_distributeEvent();

	return 0;

}
