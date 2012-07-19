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
#include <fstream>
#include "ConcreteStreamable.h"
#include "Incremental.h"
#include "ParameterScan.h"
#include "Series.h"
#include "SequenceFactory.h"
#include "SequenceIteratorIterator.h"
#include "UtilLibException.h"
#include "UtilTest.h"

using namespace UtilLib;
using namespace std;

UtilTest::UtilTest
(
	boost::shared_ptr<ostream> p	/// requires a stream  
):
LogStream(p)
{
}

bool UtilTest::Execute() 
{
	if (! this->WriteSeries() )
		return false;
	Record("WriteSeries succeeded");

	if (! this->TestSeries() )
		return false;
	Record("TestSeries succeeded");

	if (! this->WriteIncremental() )
		return false;
	Record("WriteIncremental succeeded");

	if (! this->TestIncremental() )
		return false;
	Record("TestIncremental succeeded");

	if (! this->IteratorIncrementalTest() )
		return false;
	Record("IteratorIncrementalTest succeeded");

	if (! this->IteratorSequenceTest() )
		return false;
	Record("IteratorSequenceTest succeeded");

	if (!this->IteratorIteratorTest() )
		return false;
	Record("IteratorIteratorTest succeeded");

	if (!this->TestFactory())
		return false;
	Record("TestFactory succeeded");

	if (!this->WriteParameterScan())
		return false;
	Record("WriteParameterScan succeeded");

	if (!this->TestParameterScan())
		return false;
	Record("TestParameterScan succeeded");

	if (!this->TestTagCoding() )
		return false;
	Record("TestTagcoding succeeded");

	return true;
}

bool UtilTest::WriteSeries()
{
	ofstream str("test/test.ser");
	if (! str){
		*Stream() << "Could not open test directory to write" << endl;
		return false;
	}

	str << "<Sequence><Series><kadaver 1 2 3 4 5></Series></Sequence>\n";
	str.close();
	return true;
}

bool UtilTest::WriteIncremental()
{
	ofstream str("test/test.inc");
	if (! str){
		*Stream() << "Could not open test directory to write" << endl;
		return false;
	}

	str << "<Sequence><Incremental><goeroeboeroe 1 10 0.5></Incremental></Sequence>" << endl;
	str.close();

	return true;
}

bool UtilTest::TestSeries()
{
	ifstream str("test/test.ser");

	Series series(str);

	return true;
}

bool UtilTest::TestIncremental ()
{
	ifstream str("test/test.inc");

	Incremental incremental(str);

	return true;
}

bool UtilTest::IteratorIncrementalTest()
{	
	ifstream str("test/test.inc");

	Incremental incremental(str);


	for (SequenceIterator iter = incremental.begin(); iter != incremental.end(); iter++)
		cout << *iter << endl;

	return true;
}

bool UtilTest::IteratorSequenceTest()
{
	ifstream str("test/test.ser");

	Series series(str);
	for (SequenceIterator iter = series.begin(); iter != series.end(); iter++)
		cout << *iter << endl;

	return true;
}

bool UtilTest::IteratorIteratorTest()
{
	// don't care about uniqueness of names
	SequenceIteratorIterator test(false);

	ifstream strs("test/test.ser");
	Series series1(strs);
	Series series2 = series1;

	ifstream stri("test/test.inc");
	Incremental incremental(stri);
	try {
		test.AddLoop(series1);
		test.AddLoop(incremental);
		test.AddLoop(series2);
	}
	catch (UtilLibException& ex)
	{
		// should not trigger an exception
		cerr << ex.Description() << endl;
	}

	vector<double> current_values;

	for (Index i = 0; i < test.size(); i++, test++)
	{
		cout << test.CurrentName() << ": ";
		cout << (*test)[0] << "\t";
		cout << (*test)[1] << "\t";
		cout << (*test)[2] << endl;
	}

	return true;
}

bool UtilTest::TestFactory()
{
	ifstream str("test/test.inc");

	SequenceFactory seq;

	boost::shared_ptr<Sequence> p_seq = boost::shared_ptr<Sequence>(seq.Create(str));

	return true;
}

bool UtilTest::WriteParameterScan()
{
	ofstream str("test/test.psc");

	str << "<ParameterScan>\n";
	str << "<Sequence><Incremental><mu_in_mV 10e-3 21e-3 1e-3></Incremental></Sequence>\n";
	str << "<Sequence><Series><sigma_in_mV 1e-3 2e-3 5e-3 7e-3></Series></Sequence>\n";
	str << "</ParameterScan>\n";

	return true;
}

bool UtilTest::TestParameterScan()
{
	ifstream str("test/test.psc");
	ParameterScan ps(str);


	for(SequenceIteratorIterator iter = ps.begin(); iter != ps.end(); iter++)
	{
		cout << iter.CurrentName() << ": ";
		cout << (*iter)[0] << "\t";
		cout << (*iter)[1] << endl;
	}

	return true;
}

bool UtilTest::TestTagCoding() 
{
	string tag("<kutjecola name=\"Harige Gorilla\"  value=\"Zwaarlijvig, vet en lelijk\">");
	// Note the extra quotes!
	ConcreteStreamable streamable;
	string newtag = streamable.AddAttributeToTag(tag,"type","\"Dom en Log\"");
	// now without tags to start with, just to make sure

	string tag0("<vetvlek>");
	newtag = streamable.AddAttributeToTag(tag0,"type","\"Onga bonga\"");

	return true;
}
