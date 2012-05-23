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
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include <DynamicLib/DynamicLib.h>
#include <fstream>

using DynamicLib::DelayAlgorithm;
using DynamicLib::RateAlgorithm;
using DynamicLib::DynamicNode;
using DynamicLib::EXCITATORY;
using DynamicLib::DynamicNetwork;
using DynamicLib::SimulationRunParameter;
using std::ifstream;
using std::ofstream;

BOOST_AUTO_TEST_CASE(DelayAlgorithmCreationTest){
	
	DelayAlgorithm<double> alg(5e-3);

}


BOOST_AUTO_TEST_CASE(Delaytest){

	RateAlgorithm<double> rate(5.0);

	DynamicNode<double> node_rate(rate,EXCITATORY);

	DelayAlgorithm<double> alg(5e-3);
	DynamicNode<double> node_alg(alg,EXCITATORY);

	pair<AbstractSparseNode<double,double>*, double> p;
	p.first   = &node_rate;
	p.second  = 1.0;
	node_alg.PushBackConnection(p);

	node_rate.Evolve(1e-3); // evolve once to set value
	node_alg.Evolve(1e-3);
	double r;
	r = node_alg.GetValue();

	node_alg.Evolve(3e-3);
	r = node_alg.GetValue();

	node_alg.Evolve(4e-3);
	r = node_alg.GetValue();

	node_alg.Evolve(6e-3);
	r = node_alg.GetValue();	

	node_alg.Evolve(7e-3);
	r = node_alg.GetValue();
}

BOOST_AUTO_TEST_CASE(DelayStream){

	DelayAlgorithm<double> alg(5e-3);
	ofstream ofst("delay.alg");
	if (! ofst) 
		exit(1);

	alg.ToStream(ofst);

	ofst.close();

	ifstream ifst("delay.alg");
	if (! ifst)
		exit(1);

	DelayAlgorithm<double> algin(0);
	algin.FromStream(ifst);

	ifstream ifst2("delay.alg");
	DelayAlgorithm<double> algcon(ifst2);

}

BOOST_AUTO_TEST_CASE(DelayLinearProfile){
}