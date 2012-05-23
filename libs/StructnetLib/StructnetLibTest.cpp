// Copyright (c) 2005 - 2009 Marc de Kamps
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
#include <sstream>
#include "../NetLib/NetLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../NumtoolsLib/NumtoolsTest.h"
#include "../ConnectionismLib/ConnectionismTest.h"
#include "../SparseImplementationLib/SparseImplementationTest.h"
#include "SparseLayeredNetCode.h"
#include "NodeIdPosition.h"
#include "OrientedPatternCode.h"
#include "BasicDefinitions.h"
#include "LocalDefinitions.h"
#include "ReverseDenseOverlapLinkRelation.h"
#include "StructnetLibException.h"
#include "StructnetLibTest.h"
#include "XORLinkRelation.h"
#include "FullyConnectedLinkRelation.h"

using namespace std;
using namespace ConnectionismLib;
using namespace StructnetLib;
using namespace NetLib;
using namespace NumtoolsLib;
using namespace SparseImplementationLib;
using namespace StructnetLib;

StructNetLibTest::StructNetLibTest
(
 )
{
}

StructNetLibTest::StructNetLibTest
(
	boost::shared_ptr<ostream> p_stream,
	const string& path_name
):
LogStream(p_stream),
_path_name(path_name){
}

bool StructNetLibTest::Execute(){
#ifdef NDEBUG
	*Stream() << "NDEBUG was defined for StructNetLib" << endl;
#else
	*Stream() << "NDEBUG is undefined for StrucnetLib . Do you want debug version ?" << endl;
#endif

/*	if (! GenerateStructuredNet() )
		return false;
	Record("GenerateStructuredNet succeeded");

	if (! OrientedPatternStreamTest() )
		return false;
	Record("OrientedPatternStreamingTest succeded");

	if (! OrientedTranslationTest() )
		return false;
	Record("OrientedTranslationTest succeeded");

	if (! StructuredNetTest())
		return false;
	Record("StructuredNetTest succeeded");

	if (! NodeSizeTest() )
		return false;
	Record("NodeSizeTest succeeded");

	if (! InputFromTest() )
		return false;
	Record("InputFromTest succeeded");

	if ( ! ReverseNetTest() )
		return false;
	Record ("ReverseNet Test succeeded");

	if ( ! ReverseNetFromDiskTest() )
		return false;
	Record("ReverseNetFromDiskTest succeeded");

	if ( ! FullyConnectedTest() )
		return false;
	Record("FullyConnectedTest succeeded");

	if (! this->ForwardOrderTest() )
		return false;
	Record("ForwardOrderTest succeeded");

	if (! this->ReverseOrderTest() )
		return false;
	Record("ReverseOrderTest succeeded");

	if (! this->RZOrderTest() )
		return true;
	Record("ReverseOrderTest succeeded");
*/
	if (! this->LayerIteratorTest() )
		return true;
	Record("LayerIteratorTest succeeded");

	return true;
}

bool StructNetLibTest::GenerateStructuredNet(){

	// Create a LayerDescription vector
	vector<LayerDescription> vector_description;

	vector_description.push_back(LAYER_0);
	vector_description.push_back(LAYER_1);
	vector_description.push_back(LAYER_2);
	vector_description.push_back(LAYER_3);
	vector_description.push_back(LAYER_4);

	// Create a Network topology in terms of PhysicalStructures 
	DenseOverlapLinkRelation link_relation(vector_description);

	SpatialLayeredNet<D_LayeredReversibleSparseImplementation> net(&link_relation);
	string string_directory = NETLIB_TEST_DIR;          // test directory, containing files necessary for NetLibtest
	string string_filename  = NETLIB_TEST_FWD_NET_FILE; // name of forward network test file

	string string_fullpath = string_directory + string_filename;
	ofstream stream_out(string_fullpath.c_str());

	if ( ! stream_out ){
		Record("Opening network file failed in StructNetLibTest");
		return false;
	}

	stream_out << net;

	vector<D_Pattern> vector_of_patterns = GeneratePatterns();
	StorePatternsOnDisk(vector_of_patterns);

	return true;
}

bool StructNetLibTest::StorePatternsOnDisk
(
	const vector<D_Pattern>& vector_of_patterns
) const 
{
	vector<D_Pattern>::const_iterator iter_begin = vector_of_patterns.begin();
	vector<D_Pattern>::const_iterator iter_end   = vector_of_patterns.end();
	vector<D_Pattern>::const_iterator iter;
	for(iter = iter_begin; iter != iter_end; iter++)
	{
		Index index = static_cast<Index>(iter - iter_begin);
		ostringstream stream;
		stream << NETLIB_TEST_DIR << CreateInputFileName(index) << '\0';
		ofstream stream_pat(stream.str().c_str());
		if ( ! stream_pat )
			return false;
		stream_pat << vector_of_patterns[index];
	}

	return true;
}

SpatialConnectionistNet StructNetLibTest::GenerateNet(const SigmoidParameter& par) const
{
		// Create a LayerDescription vector
	vector<LayerDescription> vector_description;

	vector_description.push_back(LAYER_0);
	vector_description.push_back(LAYER_1);
	vector_description.push_back(LAYER_2);
	vector_description.push_back(LAYER_3);
	vector_description.push_back(LAYER_4);

	// Create a Network topology in terms of PhysicalStructures 
	DenseOverlapLinkRelation link_relation(vector_description);

	SpatialLayeredNet<D_LayeredReversibleSparseImplementation> 
		net
		(
			&link_relation,
			par
		);

	// Generate Patterns on disk, a prerequisite for generating the TrainingUnits
	vector<D_Pattern> vector_of_patterns = GeneratePatterns();
	StorePatternsOnDisk(vector_of_patterns);

	// Generate the TrainingUnits, necessary for the TrainingVector
	vector<TrainingUnit<double> > vector_training_units = GenerateTestTrainingUnits();

	BackpropTrainingVector
	<
		D_LayeredReversibleSparseImplementation,
		D_LayeredReversibleSparseImplementation::WeightLayerIterator
	> vec_train(&net, NETLIB_TRAINING_PARAMETER);

	vec_train.AcceptTrainingUnitVector(vector_training_units);


	while( vec_train.ErrorValue() > NETLIB_TEST_ENERGY )
	{
		vec_train.Train();
	}

	return net;

}

bool StructNetLibTest::StructuredNetTest() const {

	//   Jogs the entire training infrastucture for backpropagation:
	//   Reads a generated structured network from file
	//   Reads patterns that are to be trained
	//   Trains the network

	// Create forward network
	// path and file names are defined in configuration.h

	ifstream stream_forward_net( (NETLIB_TEST_DIR + NETLIB_TEST_FWD_NET_FILE).c_str() );
	if ( ! stream_forward_net )
		return false;

	SpatialLayeredNet<D_LayeredReversibleSparseImplementation> net_forward(stream_forward_net);


	ofstream stream_compare( (NETLIB_TEST_DIR + "compare.net").c_str() );
	stream_compare << net_forward;

	// Generate the TrainingUnits, necessary for the TrainingVector
	vector<TrainingUnit<double> > vector_training_units = GenerateTestTrainingUnits();

	BackpropTrainingVector
	<
		D_LayeredReversibleSparseImplementation,
		D_LayeredReversibleSparseImplementation::WeightLayerIterator
	> vector_training(&net_forward, NETLIB_TRAINING_PARAMETER);

	vector_training.AcceptTrainingUnitVector(vector_training_units);
	
	while( vector_training.ErrorValue() > NETLIB_TEST_ENERGY )
		vector_training.Train();


	return true;
}

vector<TrainingUnit<double> > StructNetLibTest::GenerateTestTrainingUnits() const {
	 vector<TrainingUnit<double> > vector_return;

	 // Loop over the input patterns and create the training units
	 
	 // Loop over all input pattern files, create input pattern and associate
	 // the input vector with the correct output category

	 for (Index n_input_files = 0; n_input_files < NR_PATTERN_FILES; n_input_files++ ){
		 string name_input_file = CreateInputFileName(n_input_files);
		 string name_input_path = NETLIB_TEST_DIR + name_input_file;
		 // open file stream for input file
		 ifstream stream_input_file(name_input_path.c_str());
		 if ( ! stream_input_file )
			 throw StructnetLibException(string("Creation input file failed. Did you create a test directory ?"));
		 
		 // create and fill input pattern
		 D_Pattern pat_input;
		 stream_input_file >> pat_input;

		 // get the correct output pattern
		 D_Pattern pat_output = GetOutputPatternForThisInputFile(n_input_files);

		 // push back the resulting training unit
		 vector_return.push_back(TrainingUnit<double>(pat_input,pat_output));
	 }

	 return vector_return;
}

	string StructNetLibTest::CreateInputFileName(Index n_file) const 
	{
		ostringstream stream_output;
		stream_output << NET_LIB_TEST_PATTERN_BASE_NAME << n_file << PATTERN_EXTENSION << '\0';
		return stream_output.str();
	}

	D_Pattern StructNetLibTest::GetOutputPatternForThisInputFile(Index n_input_file) const 
	{

	// n_input_file labels the input files, therefore check if the range is correct
	assert (n_input_file < NR_PATTERN_FILES);

	// the return pattern is an output pattern:
	D_Pattern pattern_return(NR_OUTPUT_CATEGORIES);
	pattern_return.Clear();

	// patterns are grouped, every group belongs to a single output pattern

	int nr_files_per_category = NR_PATTERN_FILES/NR_OUTPUT_CATEGORIES;

	assert
	( 
	       std::div
	       (
			static_cast<int>(NR_PATTERN_FILES),
			static_cast<int>(NR_OUTPUT_CATEGORIES)
		).rem == 0
	);

	Index n_output_index = 
		static_cast<Index>
		(
			 std::div
			(
				static_cast<int>(n_input_file), 
				nr_files_per_category
			).quot
		);
	pattern_return[n_output_index] = 1;

	return pattern_return;
}

	vector<D_Pattern> StructNetLibTest::GeneratePatterns() const 
	{
		vector<D_Pattern> vector_of_patterns;

		// Create a pattern with the right size of the
		D_Pattern pattern_input(LAYER_0._nr_x_pixels*LAYER_0._nr_y_pixels*LAYER_0._nr_features);

		pattern_input.Clear();

		// Create pattern 0
		pattern_input[157] = 1;
		pattern_input[445] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 1
		pattern_input[265] = 1;
		pattern_input[553] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 2
		pattern_input[166] = 1;
		pattern_input[454] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 3
		pattern_input[274] = 1;
		pattern_input[562] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 4
		pattern_input[13] = 1;
		pattern_input[301] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();


		// Create pattern 5
		pattern_input[121] = 1;
		pattern_input[409] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 6
		pattern_input[22]  = 1;
		pattern_input[310] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 7
		pattern_input[130] = 1;
		pattern_input[418] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 8
		pattern_input[12] = 1;
		pattern_input[13] = 1;
		pattern_input[14] = 1;
		pattern_input[289] = 1;
		pattern_input[301] = 1;
		pattern_input[313] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

	
		// Create pattern 9
		pattern_input[120] = 1;
		pattern_input[121] = 1;
		pattern_input[122] = 1;
		pattern_input[397] = 1;
		pattern_input[409] = 1;
		pattern_input[421] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();
	
		// Create pattern 10
		pattern_input[21] = 1;
		pattern_input[22] = 1;
		pattern_input[23] = 1;
		pattern_input[298] = 1;
		pattern_input[310] = 1;
		pattern_input[322] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();
		
		// Create pattern 11
		pattern_input[129] = 1;
		pattern_input[130] = 1;
		pattern_input[131] = 1;
		pattern_input[406] = 1;
		pattern_input[418] = 1;
		pattern_input[430] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();
	
		// Create pattern 12
		pattern_input[144] = 1;
		pattern_input[157] = 1;
		pattern_input[170] = 1;
		pattern_input[434] = 1;
		pattern_input[445] = 1;
		pattern_input[456] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();
	
		// Create pattern 13
		pattern_input[252] = 1;
		pattern_input[265] = 1;
		pattern_input[278] = 1;
		pattern_input[542] = 1;
		pattern_input[553] = 1;
		pattern_input[564] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		// Create pattern 14
		pattern_input[153] = 1;
		pattern_input[166] = 1;
		pattern_input[179] = 1;
		pattern_input[443] = 1;
		pattern_input[454] = 1;
		pattern_input[465] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();	

		// Create pattern 15
		pattern_input[261] = 1;
		pattern_input[274] = 1;
		pattern_input[287] = 1;
		pattern_input[551] = 1;
		pattern_input[562] = 1;
		pattern_input[573] = 1;
		vector_of_patterns.push_back(pattern_input);
		pattern_input.Clear();

		return vector_of_patterns;
	}

bool StructNetLibTest::OrientedPatternStreamTest() const {
	OrientedPattern<double> opattern_test(OP_N_X,OP_N_Y,OP_N_Z);

	string file_name = NETLIB_TEST_DIR + OPATTERN_TEST_FILE;
	ofstream file_stream(file_name.c_str());
	
	opattern_test.Clear();
	opattern_test(1,1,1) = 1;

	file_stream << opattern_test;
	file_stream.close();

	OrientedPattern<double> opattern_default;
	ifstream input_file_stream(file_name.c_str());
	if ( ! input_file_stream )
		return false;
	input_file_stream >> opattern_default;
	input_file_stream.close();

	ifstream input_file_stream2(file_name.c_str());
	if ( ! input_file_stream2 )
		return false;
	OrientedPattern<double> opattern_istream(input_file_stream2);
	input_file_stream2.close();

	if (opattern_istream(1,1,1) != 1 )
		return false;

	return true;
}

bool StructNetLibTest::NodeSizeTest() const {
	// obsolete

	return true;
}

bool StructNetLibTest::InputFromTest() const
{

	SparseImplementationTest test;
	
	// Need a threshold
	LayeredArchitecture architecture(test.CreateXORArchitecture());

	LayeredNetwork<D_LayeredReversibleSparseImplementation> net(&architecture);

	if ( ! net.IsInputNeuronFrom(NodeId(1), NodeId(3)) )
		return false;

	if ( ! net.IsInputNeuronFrom(NodeId(2), NodeId(3)) )
		return false;

	if ( ! net.IsInputNeuronFrom(NodeId(1), NodeId(4)) )
		return false;

	if ( ! net.IsInputNeuronFrom(NodeId(2), NodeId(4)) )
		return false;

	if ( ! net.IsInputNeuronFrom(NodeId(3), NodeId(5)) )
		return false;

	if ( ! net.IsInputNeuronFrom(NodeId(4), NodeId(5)) )
		return false;

	if ( net.IsInputNeuronFrom(NodeId(1), NodeId(5)) )
		return false;

	if ( net.IsInputNeuronFrom(NodeId(2), NodeId(5)) )
		return false;

	return true;
}

bool StructNetLibTest::ReverseNetTest() const
{
	XORLinkRelation rel;

	SpatialConnectionistNet net(&rel);
	
	ReverseDenseOverlapLinkRelation rev(net);

	SpatialConnectionistNet net_rev(&rev);

	return true;
}

bool StructNetLibTest::ReverseNetFromDiskTest() const
{
	// Read network from disk
	// and reverse that

	ifstream stream_forward_net( (NETLIB_TEST_DIR + NETLIB_TEST_FWD_NET_FILE).c_str() );
	if ( ! stream_forward_net )
		return false;

	SpatialLayeredNet<D_LayeredReversibleSparseImplementation> net_forward(stream_forward_net);

	ReverseDenseOverlapLinkRelation reverse(net_forward);

	SpatialLayeredNet<D_LayeredReversibleSparseImplementation> net_reverse(&reverse);

	ofstream boink("boink.txt");

	boink << net_reverse;
	return true;
}

bool StructNetLibTest::OrientedTranslationTest() const
{
	D_OrientedPattern pat(10,10,4);

	pat(1,1,0) = 1.0;

	int tr = 0;
	pat.MinTrx(&tr);
	if ( tr != -1 ) 
		return false;
	pat.MaxTrx(&tr);
	if ( tr != 8 )
		return false;

	pat(8,2,3) = 1.0;

	pat.MinTrx(&tr);
	if ( tr != -1 )
		return false;
	pat.MaxTrx(&tr);
	if (tr != 1 )
		return false;

	pat.Clear();
	pat(8,6,2) = 1.0;
	// one translation in the x-rection should be possible
	if ( ! pat.TransX(1) )
		return false;
	// the pattern should be translated
	if ( pat(9,6,2) != 1.0)
		return false;

	// but a second translation should not work
	if ( pat.TransX(1) )
		return false;

	// pattern should be unchanged after last, unsuccesful attempt
	if ( pat(9,6,2) != 1.0)
		return false;


	// repeat for y
	pat.Clear();
	pat(1,1,2) = 1.0;
	pat.MinTry(&tr);
	if ( tr != -1 ) 
		return false;
	pat.MaxTry(&tr);
	if ( tr != 8 )
		return false;

	pat(2,8,3) = 1.0;

	pat.MinTry(&tr);
	if ( tr != -1 )
		return false;
	pat.MaxTry(&tr);
	if (tr != 1 )
		return false;

	pat.Clear();
	pat(6,8,2) = 1.0;
	// one translation in the x-rection should be possible
	if ( ! pat.TransY(1) )
		return false;
	// the pattern should be translated
	if ( pat(6,9,2) != 1.0)
		return false;

	// but a second translation should not work
	if ( pat.TransY(1) )
		return false;

	// pattern should be unchanged after last, unsuccesful attempt
	if ( pat(6,9,2) != 1.0)
		return false;
	return true;
}

bool StructNetLibTest::FullyConnectedTest() const
{
	FullyConnectedLinkRelation link(2,2);

	SpatialConnectionistNet net(&link);
	return true;
}

bool StructNetLibTest::ForwardOrderTest() const
{
	vector<LayerDescription> vec;
	vec.push_back(LAYER_0);
	vec.push_back(LAYER_1);
	vec.push_back(LAYER_2);

	NodeIdPosition idpos(vec);

	ForwardOrder begin = idpos.begin();
	ForwardOrder end   = idpos.end();

	NodeId id = begin.Id();
	if (id != NodeId(1))
		return false;

	begin++;
	id = begin.Id();
	if (id != NodeId(2))
		return false;

	return true;
}

bool StructNetLibTest::ReverseOrderTest() const
{
	return true;
}

bool StructNetLibTest::RZOrderTest() const
{
	// a reversed network
	vector<LayerDescription> vec;
	vec.push_back(MINI_2);
	vec.push_back(MINI_1);
	vec.push_back(MINI_0);

	NodeIdPosition idpos(vec);

	RZOrder begin = idpos.rzbegin();
	RZOrder end   = idpos.rzend();

	for ( RZOrder it = idpos.rzbegin(); it != idpos.rzend(); it++)
	{
		cout << it.Id () << "|" << it.Position() << endl;
	}
	
	return true;
}

bool StructNetLibTest::LayerIteratorTest() const
{
	// Create a LayerDescription vector
	vector<LayerDescription> vector_description;

	vector_description.push_back(MINI_0);
	vector_description.push_back(MINI_1);
	vector_description.push_back(MINI_2);

	// Create a Network topology in terms of PhysicalStructures 
	DenseOverlapLinkRelation link_relation(vector_description);

	SpatialLayeredNet<D_LayeredReversibleSparseImplementation> net(&link_relation);


	typedef SpatialLayeredNet<D_LayeredReversibleSparseImplementation>::NodeType node_type;

	if ( net.begin(0)->MyNodeId()._id_value != 1)
		return false;

	NodeId id;
	for (NodeIterator<node_type> iter = net.begin(0); iter != net.end(0); iter++)
		id = iter->MyNodeId();

	if (id != NodeId(18) )
		return false;

	if ( net.begin(1)->MyNodeId()._id_value != 19 )
		return false;

	for (NodeIterator<node_type> iter = net.begin(1); iter != net.end(1); iter++)
		id = iter->MyNodeId();

	if ( id != NodeId(22) )
		return false;

	if ( net.begin(2)->MyNodeId()._id_value != 23 )
		return false;


	return true;
}