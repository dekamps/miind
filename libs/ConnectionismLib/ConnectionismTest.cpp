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
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include <fstream>
#include "../NetLib/NetLib.h"
#include "../NetLib/NetLibTest.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "../SparseImplementationLib/SparseImplementationTest.h"
#include "BackpropTrainingVectorCode.h"
#include "HebbianCode.h"
#include "LocalDefinitions.h"
#include "LayeredNetworkCode.h"
#include "ConnectionismTest.h"
#include "ConnectionismTestParameters.h"


using namespace std;
using namespace NetLib;
using namespace SparseImplementationLib;
using namespace ConnectionismLib;
using namespace NumtoolsLib;

ConnectionismTest::ConnectionismTest():
_path_name(""){
}

ConnectionismTest::ConnectionismTest
(
	boost::shared_ptr<ostream> p_stream,
	const string& path_name
):
LogStream(p_stream),
_path_name(path_name){
}

bool ConnectionismTest::Execute(){

	if ( ! SimpleLayeredArchitectureTest() )
		return false;
	Record("SimpleLayeredArchitectureTest succeeded");

	if ( ! TestSingleXORStep() )
	  return false;
	Record("TestSingleXORStep succeeded");

	if ( ! XORTest() )
	  return false;
	Record("XOR test succeeeded");

	if ( ! HebbianTest() )
	  return false;
	Record("Hebbian test succeeeded");

	if ( ! SquashStreamTest() )
		return false;
	Record("Squash stream test succeeded");

	if (! InputFromTest() )
		return false;
	Record("InputFromTest succeeded");

	if (! TrainingUnitStreamingTest() )
		return false;
	Record("TrainingUnitStreamingTest succeeded");

	return true;
}

void ConnectionismTest::PushBackXorPatterns
	(
		BackpropTrainingVector
		<
			D_LayeredReversibleSparseImplementation,
			D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold
		>& training_vector
	) const 
{
	D_Pattern pat_input(2);
	D_Pattern pat_output(1);

	// 00 -> 0
	pat_input[0] = 0;
	pat_input[1] = 0;

	pat_output[0] = 0;
	training_vector.PushBack(TrainingUnit<double>(pat_input,pat_output));

	// 01 -> 1
	pat_input[0] = 0;
	pat_input[1] = 1;
	
	pat_output[0] = 1;
	training_vector.PushBack(TrainingUnit<double>(pat_input,pat_output));

	// 10 -> 1
	pat_input[0] = 1;
	pat_input[1] = 0;

	pat_output[0] = 1;
	training_vector.PushBack(TrainingUnit<double>(pat_input,pat_output));

	// 11 -> 0

	pat_input[0] = 1;
	pat_input[1] = 1;

	pat_output[0] = 0;
	training_vector.PushBack(TrainingUnit<double>(pat_input,pat_output));
}

bool ConnectionismTest::SimpleLayeredArchitectureTest() const
{
	vector<Layer> vec;
	vec.push_back(2);
	vec.push_back(2);
	vec.push_back(1);

	LayeredArchitecture arch(vec);

	LayeredNetwork<D_LayeredReversibleSparseImplementation> net(&arch);


	// jog the copy constructor
	LayeredNetwork<D_LayeredReversibleSparseImplementation> net2 = net;

	// jog the copy operator
	net2 = net;

	return true;
}

bool ConnectionismTest::TestValuesSingleXORStep
(
	const LayeredNetwork<D_LayeredReversibleSparseImplementation>& net
) const {
	return ( 
			IsApproximatelyEqualTo( net.GetActivity(NodeId(5)), DELTA_1, EPSILON_XOR ) &&
			IsApproximatelyEqualTo( net.GetActivity(NodeId(3)), DELTA_2, EPSILON_XOR ) &&
			IsApproximatelyEqualTo( net.GetActivity(NodeId(4)), DELTA_2, EPSILON_XOR ) 
			)  ?
	  true : false;
}

bool ConnectionismTest::TestSingleXORStep() const {

	SparseImplementationTest test;
	
	// Need a threshold
	LayeredArchitecture architecture(test.CreateXORArchitecture());

	LayeredNetwork<D_LayeredReversibleSparseImplementation> net(&architecture);

	BackpropTraining<D_LayeredReversibleSparseImplementation,D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold> 
		backprop(XOR_SINGLE_STEP_PARAMETER);
	
	// Net inserts squashing values and implementation pointer
	net.SetTraining(backprop);
	// Intialization is necessary because only now the TrainingAlgorithm has become available
	net.Initialize();

	D_Pattern pattern_in(2);
	pattern_in[0] = 0;
	pattern_in[1] = 0;

	D_Pattern pattern_out(1);
	pattern_out[0] = 1;

	TrainingUnit<double> unit_train(pattern_in, pattern_out);
	
	net.ReadIn(pattern_in);
	net.Evolve();

	net.Train(unit_train);

	return TestValuesSingleXORStep(net);
	
}

bool ConnectionismTest::XORTest() const {

	SparseImplementationTest test;
	LayeredArchitecture architecture = test.CreateXORArchitecture();

	LayeredNetwork<D_LayeredReversibleSparseImplementation> net(&architecture);

	BackpropTrainingVector
	<
		D_LayeredReversibleSparseImplementation,
		D_LayeredReversibleSparseImplementation::WeightLayerIteratorThreshold
	> vector_backprop(&net, XOR_PARAMETER);
	
	// push back the training patterns, belonging to an XOR network
	PushBackXorPatterns(vector_backprop);
	
	for (int i = 0; i < static_cast<int>(NR_XOR_TRAINING_STEPS); i++ )
	    vector_backprop.Train();

	if ( vector_backprop.ErrorValue() < XOR_TEST_ENERGY)
		return true;
	else
		return false;
}

bool ConnectionismTest::HebbianTest() const {
	SparseImplementationTest test;
	LayeredArchitecture architecture = test.CreateXORArchitecture();

	LayeredNetwork<D_LayeredReversibleSparseImplementation>  net(&architecture);

	// It needs a training parameter,
	// but the values don't matter
	HebbianTrainingParameter par_train;
	Hebbian	<D_LayeredReversibleSparseImplementation> hebbian(par_train);
	net.SetTraining(hebbian);

	net.SetActivity(NodeId(1),2);
	net.SetActivity(NodeId(2),2);
	net.SetActivity(NodeId(3),4);
	net.SetActivity(NodeId(4),4);
	net.SetActivity(NodeId(5),8);

	TrainingUnit<double> unit_empty;
	net.Initialize();
	net.Train(unit_empty);

	net.SetActivity(NodeId(1),1);
	net.SetActivity(NodeId(2),1);
	net.SetActivity(NodeId(3),-1);
	net.SetActivity(NodeId(4),-1);
	net.SetActivity(NodeId(5),1);

	net.Train(unit_empty);

	// weight 3 - 1, 3- 2, 4- 1, 4-2  should be 7
	// weight 5 -3, 5 -4 should be 31
	double f = net.GetWeight(NodeId(3),NodeId(1));

	if ( f != 7 ) 
		return false;

	f = net.GetWeight(NodeId(5),NodeId(3));
	if ( f != 31 )
		return false;

	return true;
}
bool ConnectionismTest::TrainingUnitStreamingTest() const
{
	D_Pattern pat_in(8);
	pat_in.Clear();

	pat_in[5] = pat_in[7] = 1.0;

	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[1] = 2.0;

	D_TrainingUnit tu(pat_in, pat_out);

	ofstream ost_test("test/test.tu");
	if (! ost_test )
		throw ConnectionismLibException("Couldn't open TraingUnit streaming file");

	ost_test << tu;
	ost_test.close();

	ifstream ist_test("test/test.tu");
	if (! ist_test )
		throw ConnectionismLibException("Couldn't open TraingUnit streaming file");

	D_TrainingUnit tu_st;
	tu_st.FromStream(ist_test);

	if ( tu_st.OutPat()[1] != 2.0 )
		return false;

	return true;
}

bool ConnectionismTest::InputFromTest() const
{
	return true;
}

bool ConnectionismTest::SquashStreamTest() const 
{

	Sigmoid sigmoid;

	string full_path_name = STR_TEST_DIR + STR_SQUASH_TEST;
	ofstream stream_squash(full_path_name.c_str());
	if ( ! stream_squash )
		throw NetLibException(STR_TEST_SQUASH_OPEN_FAILED);

	stream_squash << sigmoid;
	stream_squash.close();

	ifstream stream_squash_read(full_path_name.c_str());
	if ( ! stream_squash_read )
		throw NetLibException(STR_TEST_SQUASH_OPEN_FAILED);

	SquashingFunctionFactory factory;

	try
	{
		auto_ptr<AbstractSquashingFunction> p_squash(factory.FromStream(stream_squash_read));
	}
	catch (...)
	{
		return false;
	}

	return true;
}
