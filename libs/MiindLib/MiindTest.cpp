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
//      If you use this software in work leading to a scientific publication, you should cite
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include <fstream>
#include "../MiindLib/MiindLib.h"
#include "MiindTest.h"
#include "XMLRunParameter.h"

using namespace MiindLib;
using namespace std;

MiindTest::MiindTest(boost::shared_ptr<ostream> p):
LogStream (p)
{
}
MiindTest::~MiindTest()
{
}

bool MiindTest::Execute() 
{
	if (! this->XMLNodeStreamingTest() )
		return false;
	Record("XMLNodeStreamingTest succeeded");

	if (! this->XMLConnectionStreamingTest() )
		return false;
	Record("XMLConnectionStreamingTest");

	if (! this->XMLRunParameterStreamingTest() )
		return false;
	Record("XMLRunParameterStreamingTest succeeded");

	return true;
}


bool MiindTest::XMLNodeStreamingTest() const
{
	ofstream ofst("test/bla.node");
	if (! ofst){
		cout << "Could open node file. Is there a test directory?" << endl;;
		return false;
	}
	MiindLib::XMLNode node("EXCITATORY", "Cortical Background","CBG_ALG");
	node.ToStream(ofst);
	ofst.close();

	ifstream ifs("test/bla.node");
	XMLNode read_node(ifs);

	ofstream stnew("test/new.node");
	read_node.ToStream(stnew);

	// the text in "new.node" and "bla.node" should be identical

	return true;
}

bool MiindTest::XMLConnectionStreamingTest() const
{
	PopulationConnection con(10,0.01);
	XMLConnection<PopulationConnection> xcon("indy","outty",con);
	ofstream ofst("test/bla.con");
	if (! ofst){
		cerr << "Please open a test directory" << endl;
		return false;
	}
	xcon.ToStream(ofst);
	ofst.close();

	ifstream ifst("test/bla.con");
	XMLConnection<PopulationConnection> icon(ifst);

	ofstream ofnew("test/new.con");
	icon.ToStream(ofnew);

	return true;
}

bool MiindTest::XMLRunParameterStreamingTest() const
{
	XMLRunParameter par_out("babagaab.root",true,true,false);
	par_out.AddNodeToCanvas("LIF E");
	par_out.AddNodeToCanvas("LIF I");

	ofstream ofst("test/babagaab.par");
	if (! ofst )
		return false;

	par_out.ToStream(ofst);

	ofst.close();

	ifstream ifst("test/babagaab.par");
	if (! ifst )
		return false;

	XMLRunParameter par_in(ifst);

	ofstream ofst_new("test/par_run_new.par");
	par_in.ToStream(ofst_new);

	return true;
}