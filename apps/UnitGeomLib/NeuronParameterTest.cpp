// Copyright (c) 2005 - 2014 Marc de Kamps
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

#include <boost/test/unit_test.hpp>
#include <boost/test/execution_monitor.hpp>
#include <GeomLib.hpp>


using namespace GeomLib;
using MPILib::Time;


BOOST_AUTO_TEST_CASE(NeuronParameterTest ) {

	Potential theta      = 20e-3;
	Potential V_reset    = 0.0;
	Potential V_reversal = 0.0;
	Time      t_ref      = 3e-3;
	Time      tau        = 20e-3;

	NeuronParameter par_neuron(theta, V_reset, V_reversal, t_ref, tau);
}

BOOST_AUTO_TEST_CASE( NeuronWrongOrderTest ){

	Potential theta      = 20e-3;
	Potential V_reset    = 0.0;
	Potential V_reversal = 0.0;
	Time      t_ref      = 3e-3;
	Time      tau        = 20e-3;

	BOOST_CHECK_THROW(NeuronParameter par_neuron_wrong(V_reset, theta, V_reversal, t_ref,tau), GeomLibException);
	BOOST_CHECK_THROW(NeuronParameter par_neuron_wrong_t(V_reversal, V_reset, theta, tau, t_ref),GeomLibException);
}

