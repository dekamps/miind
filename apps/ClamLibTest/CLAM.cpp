// Copyright (c) 2005 - 2010 Marc de Kamps
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
//      the 'currently valid reference', which can be found at http://clam.sourceforge.net

#ifdef WIN32
#pragma warning (disable: 4267)
#pragma warning (disable: 4996)
#endif

#include <iostream>

using std::cout;
using std::endl;

#include <ClamLib/ClamLibTest.h>
#include <ConnectionismLib/ConnectionismTest.h>
#include <StructnetLib/StructnetLibTest.h>

using DynamicLib::D_DynamicNetwork;
using DynamicLib::D_DynamicNode;
using ConnectionismLib::ConnectionismTest;
using StructnetLib::OrientedPattern;

using ClamLib::ClamLibTest;

int main(int argc, char* argv[])
{
	cout << "CLAM !!" << endl;

	boost::shared_ptr<ostream> p(new ofstream("muggawugga"));
	ClamLibTest test(p);
	try 
	{
		if ( test.Execute() )
			cout << "Ran fine" << endl;
		else
			cout << "Test failed" << endl;
	}

	catch (UtilLib::GeneralException& excep)
	{
		cout << "Util exception"   << endl;
		cout << excep.Description() << endl;
		cout << "Is there a test directory?" << endl;
		return false;
	}
	catch (...)
	{
		cout << "Some error occured" << endl;
		return false;
	}

	return 0;
}

