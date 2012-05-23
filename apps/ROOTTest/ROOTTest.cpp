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
#include <iostream>
#include <fstream>
#include <TFile.h>
#include <TROOT.h>
#include <TApplication.h>
#include <TClassTable.h>
#include <RootLayerDescription.h>
#include <RootLayeredNetDescription.h>


using ClamLib::RootLayerDescription;
using ClamLib::RootLayeredNetDescription;


using std::cout;
using std::endl;
using std::ofstream;

int main(int argc, char* argv[])
{
	try {

	  ofstream ofst("test/test.txt");
	  if (! ofst){
	    cout << "Please creat a directory called test" << endl;
	    exit(0);
	  }
		RootLayerDescription desc;
		desc._nr_x_pixels = 787;
		desc._nr_y_pixels = 10;
		desc.SetName("goeroeboe");

		TFile file("test/rootlayer.root","RECREATE");
		desc.Write();
		file.Close();

		TFile file2("test/rootlayer.root");
		RootLayerDescription* p_desc = (RootLayerDescription*)file2.Get("goeroeboe");
		cout << "Read in: " << p_desc->GetName() << endl;

		// write out a vector of RootLayerDescription


	 	if (! TClassTable::GetDict("vector<float>") )
			gROOT->ProcessLine("#include <vector>");

		vector<RootLayerDescription> vec_desc;
		RootLayerDescription desc1,desc2;
		desc1._nr_features = 101;
		desc2._nr_features = 77;

		vec_desc.push_back(desc1);
		vec_desc.push_back(desc2);

		RootLayeredNetDescription descvec(vec_desc);
		TFile file3("test/rootlayervector.root","RECREATE");
		descvec.SetName("vector_test");
		descvec.Write();
		file3.Close();

	}

	catch (...)
	{
		cout << "Unknown error occured" << endl;
	}

	cout << "Finished" << endl;
	return 0;
}
