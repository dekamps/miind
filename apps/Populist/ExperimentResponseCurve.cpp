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

#include <sstream>
#include <PopulistLib/PopulistLib.h>
#include <UtilLib/UtilTest.h>
#include <DynamicLib/DynamicLib.h>
#include <PopulistLib/PopulistLib.h>
#include <PopulistLib/TestPopulistCode.h>
#include <MiindLib/MiindLib.h>

using DynamicLib::DynamicLibException;
using DynamicLib::RootReportHandler;
using PopulistLib::CovertMuSigma;
using PopulistLib::SinglePopulationInput;
using std::ostringstream;

const string METAFILENAME("test/responsecurve.meta");

int main(int argc, char* argv[])
{
	
	ifstream istr(METAFILENAME.c_str());
	if (! istr){
		ofstream ostr(METAFILENAME.c_str());
		if (! ostr)
			cout << "Can't handle the meta file. Does the directory 'test' exist?" << endl;
	}
	
	ParseResponseCurveMetaFile parser(METAFILENAME);
	PopulationParameter			par_pop  = parser.ParPop();
	PopulistSpecificParameter	par_spec = parser.ParSpec();

	boost::shared_ptr<ostream> p(new ostringstream); 
	TestPopulist test(p);
	for
	(
		SequenceIteratorIterator iter = parser.ParScan().begin();
		iter != parser.ParScan().end();
		iter++
	){
		cout << iter.CurrentName() << endl;
		double mu    = (*iter)[0];
		double sigma = (*iter)[1];

		string name("test/single");
		string full_name = name + iter.CurrentName();
		string log_name = string(full_name.c_str()) + string(".log");

		RootReportHandler handler(full_name,false,true);
		handler.SetPotentialRange(0.0,par_pop._theta);
		handler.SetTimeRange(0.0,1.0);


		SimulationRunParameter par_sim = 
			parser.ParSim
			(
				handler,
				log_name
			);
		SinglePopulationInput inp = PopulistLib::CovertMuSigma(mu,sigma,par_pop);

		try {
		  test.GenericOnePopulationTest(inp._rate,inp._h,par_pop,par_sim,par_spec,true);
		}
		catch(DynamicLibException& excep){
			cout << excep.Description() << endl;
		}
	}

	return 0;
}

