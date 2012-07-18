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
#include <MPILib/include/utilities/LogStream.hpp>
#undef protected
#undef private
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>


#include <boost/test/minimal.hpp>
using namespace boost::unit_test;
using namespace MPILib::utilities;


void test_Constructor() {

	LogStream ls;

	BOOST_CHECK(ls._isTimeAvailable == true);

	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls2(os);
	BOOST_CHECK(ls2._isTimeAvailable == true);
	BOOST_CHECK(ls2._pStreamLog == os);

}

void test_Destructor() {
	std::shared_ptr<std::ostream> os1(new std::ostringstream);
	LogStream* ls3 = new LogStream(os1);
	delete ls3;

	std::stringstream temp;
	temp << "Total time: ";
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os1)->str().compare(0, temp.str().length(), temp.str())==0);
}

void test_Record() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	ls.record("blub");
	std::string s = "blub";
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s)!= std::string::npos);
}

void test_GetStream() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(ls.getStream()==os);
}

void test_OpenStream() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(ls.openStream(os)==false);

	LogStream ls1;
	BOOST_CHECK(ls1.openStream(os)==true);
}

void test_Operators() {
	std::shared_ptr<std::ostream> os(new std::ostringstream);
	LogStream ls(os);
	std::stringstream s;
	s << std::string("blub");
	ls << std::string("blub");
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);
	s.clear();
	s << "blub2";
	ls << "blub2";
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);
	s.clear();
	s << int(1);
	ls << int(1);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);
	s.clear();
	s << double(1.4);
	ls << double(1.4);
	//cast needed as otherwise str() is not available
	BOOST_CHECK(
			std::dynamic_pointer_cast<std::ostringstream>(os)->str().find(s.str())!= std::string::npos);

}

int test_main(int argc, char* argv[]) // note the name!
		{
	test_Constructor();
	test_Destructor();
	test_Record();
	test_GetStream();
	test_OpenStream();
	test_Operators();

	return 0;

}

