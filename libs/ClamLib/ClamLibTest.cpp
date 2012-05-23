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
#ifdef WIN32
#pragma warning (disable:4244)
#pragma warning (disable:4267)
#pragma warning (disable:4996)
#endif 

#include <TFile.h>
#include <TROOT.h>
#include <TApplication.h>
#include <TClassTable.h>
#include <boost/shared_ptr.hpp>
#include "ClamLibTest.h"
#include "../DynamicLib/DynamicLib.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../StructnetLib/StructnetLibTest.h"
#include "AddTrainedNetToDynamicNetwork.h"
#include "ClamLibException.h"
#include "ClamLibPositions.h"
#include "ConfigurableCreator.h"
#include "CreateDevelopingNetworks.h"
#include "CreatorStockObjects.h"
#include "CircuitDescription.h"
#include "DynamicSubNetwork.h"
#include "HomeostaticSmooth.h"
#include "JocnPattern.h"
#include "LocalDefinitions.h"
#include "PerceptronCircuitCreator.h"
#include "PerceptronConfigurableCreator.h"
#include "PerceptronOffsets.h"
#include "RootConversions.h"
#include "RootLayerDescription.h"
#include "SemiSigmoid.h"
#include "SimpleCircuitCreator.h"
#include "SimulationInfoBlock.h"
#include "SimulationInfoBlockVector.h"
#include "SimulationOrganizer.h"
#include "SimulationResult.h"
#include "SpatPosFromPhysPos.h"
#include "TestDefinitions.h"
#include "ToRootLayerDescription.h"
#include "TrainedNet.h"
#include "WeightedLink.h"

using namespace std;
using namespace StructnetLib;
using namespace ClamLib;

using DynamicLib::AbstractAlgorithm;
using DynamicLib::AbstractReportHandler;
using DynamicLib::AlgorithmBuilder;
using DynamicLib::AsciiReportHandler;
using DynamicLib::D_DynamicNetwork;
using DynamicLib::D_RateAlgorithm;
using DynamicLib::RootHighThroughputHandler;
using DynamicLib::RootReportHandler;
using DynamicLib::WilsonCowanAlgorithm;
using StructnetLib::SpatialLayeredNet;
using UtilLib::Number;
ClamLibTest::ClamLibTest
(
	boost::shared_ptr<ostream> p_stream,
	const string& path
):
LogStream(p_stream)
{
}

bool ClamLibTest::Execute() 
{
/*	if (! SemiSigmoidWriteTest() ) 
		return false;
	Record("SemiSigmoidWriteTest succeeded");

	if (! SemiSigmoidReadTest() )
		return false;
	Record("SemiSigmoidReadTest succeeded");

	if (! SemiSigmoidBuildTest() )
		return false;
	Record("SemiSigmoidBuildTest succeeded");

	if (! SmallNetPositiveConversionTest() )
		return false;
	Record("SmallNetPositiveConversionTest succeeded");

	if (! SmallNetNegativeConversionTest() )
		return false;
	Record("SmallNetNegativeConversionTest succeeded");

	if (! PositiveConversionTest() )
		return false;
	Record("PositiveConversionTest succeeded");

	if (! NegativeConversionTest() )
		return false;
	Record("NegativeConversionTest succeeded");

	if (! TrainedNetStreamingTest () )
		return false;
	Record("TrainedNetStreamingTest succeeded");

	if (! NetworkDevelopmentTest() )
		return false;
	Record("NetworkDevelopmentTest succeeded");

	if (! TestNetworkDevelopmentTest() )
		return false;
	Record("TestNetworkDevelopmentTest failed");

	if (! CheckReverseNetTraining() )
		return false;
	Record("CheckReverseNetTraining succeeded");
*/
	if (! JOCNFWDNetConversionTest() )
		return false;
	Record("JOCNFWDNetConversionTest succeeded");
/*
	if (! JOCNFunctorTest() )
		return false;
	Record("JOCNFunctorTest succeeded");

	if (! JOCNConversionTest() )
		return false;
	Record("JOCNConversionTest succeeded");

	if (! JOCNCorrespondenceTest() )
		return false;
	Record("JOCNCorrespondenceTest succeeded");

	if ( ! RootWriteLayerTest() )
		return false;
	Record("RootWriteLayerTest succeeded");

	if ( ! RootWriteLayerVectorTest() )
		return false;
	Record("RootWriteLayerVectorTest succeeded");

	if ( ! IdWriteTest() )
		return false;
	Record("IdWriteTest succeeded");

	if ( ! WeightLinkWriteTest() )
		return false;
	Record("WeightLinkWriteTest");

	if (! this->CircuitInfoReadTest() )
		return false;
	Record("CircuitInfoReadTest");

	if (! this->CircuitInfoWriteTest() )
		return false;
	Record("CircuitInfoWriteTest");

	if (! this->CircuitCreatorProxyTest() )
		return false;
	Record("CircuitCreatorProxyTest succeeded");

	if (! this->SmallNetMetaTest() )
		return false;
	Record("SmallNetMetaTest succeeded");

	if (! this->SimulationInfoBlockWriteTest() )
		return false;
	Record("SimulationInfoBlockWriteTest succeeded");

	if (! this->SimulationInfoBlockReadTest() )
		return false;
	Record("SimulationInfoBlockReadTest succeeded");

	if (! this->CircuitFactoryTest() )
		return false;
	Record("CircuitFactoryTest succeded");

	if (! this->SmallPositiveInfoTest() ) 
		return false;
	Record("SmallPositiveInfoTest succeeded");

	if (! this->SimulationInfoBlockVectorWriteTest() )
		return false;
	Record("SimulationInfoBlockVectorWriteTest");

	if (! this->SimulationInfoBlockVectorReadTest() )
		return false;
	Record("SimulationInfoBlockVectorReadTest");

	if (! this->SimulationOrganizerSmallDirectTest() )
		return false;
	Record("SimulationOrganizerSmallDirectTest succeeded");

	if (! this->SimulationInfoJOCNFFDTest() )
		return false;
	Record("SimulationInfoJOCNFFDTest");

	if (! this->SimulationInfoJOCNTest() )
		return false;
	Record("SimulationInfoJOCNTest");

	if (! this->SubNetworkIteratorTest() )
		return false;
	Record("SubNetworkIteratorTest succeeded");

	if (!this->ReverseSubNetworkIteratorTest() )
		return false;
	Record("ReverseSubNetworkIteratorTest succeeded");

	if (! this->SimulationResultIteratorTest() )
		return false;
	Record("SimulationResultIteratorTest succeeded");

	if (! this->IndexWeightSerializationTest() )
		return false;
	Record("IndexWeightSerializationTest succeeded");

	if (! this->CircuitNodeRoleSerializationTest())
		return false;
	Record("CircuitNodeRoleSerializationTest succeeded");

	if (! this->ConfigurableCreatorTest() )
		return false;
	Record("ConfigurableCreatorTest succeeded");

	if (! this->PerceptronConfigurableCreatorTest() )
		return false;
	Record("PerceptronConfigurableCreatorTest succeeded");

	if (! this->SimulationInfoJOCNFFDConfigurableTest() )
		return false;
	Record("SimulationInfoJOCNFFDConfigurableTest");

	if (! this->SimulationInfoJOCNConfigurableTest() )
		return false;
	Record("SimulationInfoJOCNConfigurableTest");

	if (! this->TestStockDescriptions() )
		return false;
	Record("TestStockDescriptions succeeded");

	if (! this->DynamicSubLayeredIteratorTest() )
		return false;
	Record("DynamicSubLayeredIteratorTest succeeded");

	if (! this->DynamicSubLayeredReverseIteratorTest() )
		return false;
	Record("DynamicSubLayeredReverseIteratorTest succeeded");

	if (! this->JOCNIteratorTest() )
		return false;
	Record("JOCNIteratorTest succeeded");

	if (! this->DisInhibitionTest() )
		return false;
	Record("DisinhibitionTest succeeded");
*/
	return true;
}

bool ClamLibTest::SemiSigmoidWriteTest() const
{
	WilsonCowanParameter par(10e-3,500,1.0);
	SemiSigmoid alg(par);

	ofstream s("test/semisig.alg");
	if (! s) {
		cout << "Couldn't write algorithm. Is there a test directory?" << endl;
		return false;
	}
	alg.ToStream(s);

	return true;
}

bool ClamLibTest::SemiSigmoidReadTest() const
{
	ifstream s("test/semisig.alg");
	if (! s){
		cout << "Couldn't open algorithm for read" << endl;
		return false;
	}
			
	SemiSigmoid sem(s);

	return true;
}

bool ClamLibTest::SemiSigmoidBuildTest() const
{
	ifstream s("test/semisig.alg");
	if (! s){
		cout << "Couldn't open algorithm for read" << endl;
		return false;
	}
	AlgorithmBuilder<double> builder;
	boost::shared_ptr<AbstractAlgorithm<double> > p = builder.Build(s);

	return true;
}

namespace {
	DynamicLib::Rate FeedbackRate(Time t)
	{
		return (t > 0.5) ? 5*MAX_LINEAR_STATIC_RATE : 0;
	}
}

TrainedNet ClamLibTest::GenerateSmallNet(const ClamLibTest::ConversionMode& mode) const
{
	// Purpose: Generate a very small trained feedfoward connectionist network
	// for testing purposes. In particular the conversion to a Dynamic network can be studied quickly.
	// Depending on mode a direct conversion (squasing function [0,1]) or a circuit conversion (squashing function [-1,1])
	// will be selected.
	// Author: Marc de Kamps
	// Date: 03-06-2008

	FullyConnectedLinkRelation link(2,2);
	SigmoidParameter param_sigmoid = (mode == DIRECT) ? SigmoidParameter(0.0,1.0,1.0) : SigmoidParameter(-1.0,1.0,1.0);

	SpatialConnectionistNet net(&link,param_sigmoid);

	// setting weights by hand is a bit of a pain:
	SCNodeIterator iter = net.begin() + 2;

	SCPDIterator iter_weight = iter->begin();
	iter_weight.SetWeight(1.0);
	iter_weight++;
	iter_weight.SetWeight(1.0);


	iter++;
	iter_weight = iter->begin();
	iter_weight.SetWeight(2.0);
	iter_weight++;
	iter_weight.SetWeight(-2.0);


    D_Pattern in(2);
	in[0] = 1.0;
	in[1] = 1.0;

	
	D_Pattern dummy(2);
	D_TrainingUnit tu(in,dummy);
	vector<D_TrainingUnit> vec_unit;
	vec_unit.push_back(tu);

	TrainedNet tn(net,vec_unit);
	return tn;
}

TrainedNet ClamLibTest::GenerateTrainedNet(const ClamLibTest::ConversionMode& mode) const
{
	// Purpose: Generate a trained feedfoward connectionist network
	// for testing purposes.
	// Author: Marc de Kamps
	// Date: 28-09-2006


	LayerDescription desc_1;
	desc_1._nr_features = 1;
	desc_1._nr_x_pixels = 3;
	desc_1._nr_y_pixels = 3;
	desc_1._nr_x_skips  = 1;
	desc_1._nr_y_skips  = 1;
	desc_1._size_receptive_field_x = 1;
	desc_1._size_receptive_field_y = 1;

	LayerDescription desc_2;
	desc_2._nr_features = 1;
	desc_2._nr_x_pixels = 2;
	desc_2._nr_y_pixels = 2;
	desc_2._nr_x_skips  = 1;
	desc_2._nr_y_skips  = 1;
	desc_2._size_receptive_field_x = 2;
	desc_2._size_receptive_field_y = 2;

	LayerDescription desc_3;
	desc_3._nr_features = 2;
	desc_3._nr_x_pixels = 1;
	desc_3._nr_y_pixels = 1;
	desc_3._nr_x_skips  = 1;
	desc_3._nr_y_skips  = 1;
	desc_3._size_receptive_field_x = 2;
	desc_3._size_receptive_field_y = 2;

	vector<LayerDescription> vec_desc;
	vec_desc.push_back(desc_1);
	vec_desc.push_back(desc_2);
	vec_desc.push_back(desc_3);

	DenseOverlapLinkRelation dense(vec_desc);

	SigmoidParameter param_sigmoid;
	if ( mode == DIRECT)
		param_sigmoid = SigmoidParameter(0.0,1.0,1.0);
	else
		if ( mode == CIRCUIT)
			param_sigmoid = SigmoidParameter(-1.0,1.0,1.0);
		else
			throw ClamLibException("Unknown mode");

	SpatialConnectionistNet net(&dense,param_sigmoid);

	double high_act = MAX_LINEAR_STATIC_RATE;
	double low_act  = 0.0;

	D_Pattern pat1(9);
	pat1[0] = high_act;
	pat1[1] = high_act;
	pat1[2] = 0;
	pat1[3] = 0;
	pat1[4] = 0;
	pat1[5] = 0;
	pat1[6] = 0;
	pat1[7] = 0;
	pat1[8] = 0;
	

	D_Pattern pat2(9);
	pat2[0] = 0;
	pat2[1] = high_act;
	pat2[2] = 0;
	pat2[3] = high_act;
	pat2[4] = 0;
	pat2[5] = 0;
	pat2[6] = 0;
	pat2[7] = 0;
	pat2[8] = 0;


	D_Pattern pat_out1(2);
	pat_out1[0] = low_act;
	pat_out1[1] = high_act;

	D_Pattern pat_out2(2);
	pat_out2[0] = high_act;
	pat_out2[1] = low_act;

	vector<D_TrainingUnit> vec_unit;
	vec_unit.push_back(D_TrainingUnit(pat1,pat_out1));
	vec_unit.push_back(D_TrainingUnit(pat2,pat_out2));

	TrainingParameter par;
	par._f_bias          = 0.01;
	par._f_momentum      = 0.0;
	par._f_sigma         = 0.03;
	par._f_stepsize      = 0.1;
	par._train_threshold = false;
	par._n_init          = 1;
	
	SpatialConnectionistTrainingVector vec(&net,par);
	vec.AcceptTrainingUnitVector(vec_unit);


	while (vec.ErrorValue() > 1e-5)
	{
		vec.Train();
	}

	net.ReadIn(pat1);
	net.Evolve();

	if (fabs( net.ReadOut()[0] - low_act ) > 0.01 || fabs( net.ReadOut()[1] - high_act) > 0.01)
		throw ClamLibException(STR_TRAINING_FAILED);

	TrainedNet trained(net, vec_unit);
	return trained;

}
bool ClamLibTest::SmallNetNegativeConversionTest() const
{

	TrainedNet tn = this->GenerateSmallNet(CIRCUIT);

	D_Pattern in(2);
	in[0] = 0.01;
	in[1] = 0.01;

	D_Pattern dummy(2);
	tn._vec_pattern[0] = D_TrainingUnit(in,dummy);

	D_DynamicNetwork d_net11;
	this->GenerateConversionTest
	(
		CIRCUIT, 
		tn, 
		&d_net11
	);

	string str_absname11("test/smallnegativeconversion11.txt");
	AsciiReportHandler handler11(str_absname11);

	SimulationRunParameter 
		par_run
		(
			handler11,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/smallnegativeconversion11.txt"
		);

	d_net11.ConfigureSimulation(par_run);
	d_net11.Evolve();

	ofstream strdyn("test/smallnegativeconversion11.dynnet");
	strdyn << d_net11;
	return true;
}

bool ClamLibTest::SmallNetPositiveConversionTest() const
{
	// Purpose: A net of two layers, of two nodes ech is generated. Weights are set by hand:
	//			w_31= w_32 = 1.0; w_41 = -w_42 = 2.0. Inputs 00 and inputs 11 are simulated. A positive
	//			squashing function is used so steady state values for the output nodes are 1/(1 +e^-2) = 0.8807 and
	//			1/(1 + e^-0) = 0.5, respectively for inputs 11 and 0.5 for inputs 00.
	//			The networks are written by an AsciiReportHandler, so that the simulation results can be inspected in a text editor.
	//			The only problem with this is that there is no automated way of checking the outcome of the simulations, that is
	//			why in SmallNetPositiveConversionRootTest the same test is carried out with a RootReportHandler
	// Author: Marc de Kamps
	TrainedNet tn = this->GenerateSmallNet(DIRECT);
	ofstream str_net("test/smallpositiveconversion11.net");
	if (! str_net)
		return false;
	str_net << tn._net;

	D_Pattern in(2);
	in[0] = 1.0;
	in[1] = 1.0;

	D_Pattern dummy(2);
	tn._vec_pattern[0] = D_TrainingUnit(in,dummy);

	D_DynamicNetwork d_net11;
	this->GenerateConversionTest
	(
		DIRECT, 
		tn, 
		&d_net11
	);

	string str_absname11("test/smallpositiveconversion11.txt");
	AsciiReportHandler handler11(str_absname11);

	SimulationRunParameter 
		par_run
		(
			handler11,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/smallpositiveconversion11.txt"
		);

	d_net11.ConfigureSimulation(par_run);
	d_net11.Evolve();

	ofstream str_dyn("test/smallpositiveconversion11.dynnet");
	str_dyn << d_net11;
 
	in[0] = 0;
	in[1] = 0;

	tn._vec_pattern[0] = D_TrainingUnit(in,dummy);

	D_DynamicNetwork d_net00;
	this->GenerateConversionTest
	(
		DIRECT, 
		tn, 
		&d_net00
	);

	string str_absname00("test/smallpositiveconversion00.txt");
	AsciiReportHandler handler00(str_absname00);

	SimulationRunParameter 
		par_run00
		(
			handler00,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/smallpositiveconversion00.txt"
		);

	d_net00.ConfigureSimulation(par_run00);
	d_net00.Evolve();

	return true;
}

bool ClamLibTest::PositiveConversionTest() const
{
	// Purpose: Generate a DynamicNetwork from a trained BioNetwork.
	//			This network has a positive squashing function for all its nodes,
	//			therefore the conversion is 1-1
	// Author:  Marc de Kamps
	// Date:    27-09-2006


	TrainedNet network = this->GenerateTrainedNet(DIRECT);
	string str_netname("test/positiveconversion.net");
	ofstream str_net(str_netname.c_str());
	if (! str_net)
		return false;

	str_net << network._net;

		// feed one of the input patterns, evolve and see if the output matches expectations

	string str_absname("test/positivivecoversion.txt");

	D_DynamicNetwork d_net;
	this->GenerateConversionTest
	(
		DIRECT, 
		network, 
		&d_net
	);


	AsciiReportHandler handler(str_absname);

	SimulationRunParameter 
		par_run
		(
			handler,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/positivivecoversion.txt"
		);

	d_net.ConfigureSimulation(par_run);
	d_net.Evolve();

	return true;
}


bool ClamLibTest::NegativeConversionTest() const
{
	// Purpose: Generate a DynamicNetwork from a trained BioNetwork.
	//			This network has an anti-symmetric squashing function for all its nodes,
	//			therefore the conversion is to a circuit
	// Author:  Marc de Kamps
	// Date:    19-12-2006

	// generate the network:
	TrainedNet network = this->GenerateTrainedNet(CIRCUIT);

	// feed one of the input patterns, evolve and see if the output matches expectations
	string str_absname("test/negativeconversion.txt");

	D_DynamicNetwork d_net;
	this->GenerateConversionTest
	(
		CIRCUIT, 
		network, 
		&d_net
	);

	AsciiReportHandler handler(str_absname);

	SimulationRunParameter 
		par_run
		(
			handler,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/negativeconversion.txt"
		);

	d_net.ConfigureSimulation(par_run);
	d_net.Evolve();

	return true;
}

bool ClamLibTest::GenerateConversionTest
(
		const ClamLibTest::ConversionMode&	mode,
		const TrainedNet&					network,
		D_DynamicNetwork*					p_dnet
) const
{
	// Purpose: Generate a DynamicNetwork from a trained SparseLayeredBetwork.
	//			The type of network is determined by the mode
	// Author:  Marc de Kamps
	// Date:    27-09-2006


	// create an input field so that the initial conditions of the network can be set.
	// the rate functions of the network will be read from this pattern
	OrientedPattern<Rate> 
		input_field
		(
			network._net.Dimensions()[0]._nr_x_pixels,
			network._net.Dimensions()[0]._nr_y_pixels,
			network._net.Dimensions()[0]._nr_features
		);

	D_Pattern pat_trained = network._vec_pattern[0].InPat();
    SpatialConnectionistNet net = network._net;
	for ( Index i = 0; i < pat_trained.Size(); i++ )
		input_field[i] = pat_trained[i];


	if (mode == DIRECT)
		p_dnet->SetDalesLaw(false);

	WilsonCowanParameter param_exc(20e-3,1,1);
	WilsonCowanParameter param_inh(10e-3,1,1);

	boost::shared_ptr<D_AbstractAlgorithm> p_exc;
	boost::shared_ptr<D_AbstractAlgorithm> p_inh;
	
	boost::shared_ptr<AbstractCircuitCreator> p_crea;
	if ( mode == DIRECT)
	{
		p_exc  = boost::shared_ptr<WilsonCowanAlgorithm>(new WilsonCowanAlgorithm(param_exc));
		p_inh  = boost::shared_ptr<WilsonCowanAlgorithm>(new WilsonCowanAlgorithm(param_inh));
		p_crea = boost::shared_ptr<AbstractCircuitCreator>(new SimpleCircuitCreator(p_exc.get(),p_inh.get(),p_dnet,NO_OFFSET));
	}
	else
	{
		p_exc  = boost::shared_ptr<SemiSigmoid>(new SemiSigmoid(param_exc));
		p_inh  = boost::shared_ptr<SemiSigmoid>(new SemiSigmoid(param_inh));
		p_crea = boost::shared_ptr<AbstractCircuitCreator>(new PerceptronCircuitCreator(p_exc.get(),p_inh.get(),p_dnet,PERC_OFFSET));
	}

	AddTNToDN convert;

	convert.Convert
	(
		network,
		input_field,
		*p_crea,
		p_dnet
	);

	return true;
}

bool ClamLibTest::NetworkDevelopmentTest() const
{
	string path("test");
	string str_forward(""); // the name of the generated forward file will be in this string
	string str_reverse(""); // reverse file same

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		path,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	// now that the networks have been created on disk, the second call should be a no-op (apart from testing the existing of the files on disk)
	CreateDevelopingNetworks
	(
		path,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	return true;
}

bool ClamLibTest::TestNetworkDevelopmentTest() const
{
	// Here we test whether training has been successful
	string path("test");
	string str_forward(""); // the name of the generated forward file will be in this string
	string str_reverse(""); // reverse file same

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		path,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	// The network are trained such that maximum input and output activation rates are given
	// by: MAX_LINEAR_STATIC_RATE

	// put a square in the ffd network and also in the feedback network
	D_OrientedPattern pat_in = JOCNPattern(SQUARE);
	pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);

	p_tn->_net.ReadIn(pat_in);
	p_tn->_net.Evolve();

	if ( !
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[SQUARE], MAX_LINEAR_STATIC_RATE,0.005)		&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[DIAMOND], 0, 0.005)						&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_HORIZONTAL], 0, 0.005)				&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_DIAGONAL], 0, 0.005)
		)
		return false;

	pat_in = JOCNPattern(CROSS_HORIZONTAL);
	pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);

	p_tn->_net.ReadIn(pat_in);
	p_tn->_net.Evolve();


	if ( !
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[SQUARE], 0,0.005)									&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[DIAMOND], 0, 0.005)								&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_HORIZONTAL], MAX_LINEAR_STATIC_RATE, 0.005)	&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_DIAGONAL], 0, 0.005)
		)
		return false;

	pat_in = JOCNPattern(DIAMOND);
	pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);

	p_tn->_net.ReadIn(pat_in);
	p_tn->_net.Evolve();

	if ( !
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[SQUARE], 0,0.005)									&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[DIAMOND], MAX_LINEAR_STATIC_RATE, 0.005)			&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_HORIZONTAL], 0, 0.005)						&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_DIAGONAL], 0, 0.005)
		)
		return false;

	pat_in = JOCNPattern(CROSS_DIAGONAL);
	pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);

	p_tn->_net.ReadIn(pat_in);
	p_tn->_net.Evolve();

	if ( !
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[SQUARE], 0,0.005)									&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[DIAMOND], 0, 0.005)								&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_HORIZONTAL], MAX_LINEAR_STATIC_RATE, 0.005)	&&
		NumtoolsLib::IsApproximatelyEqualTo(p_tn->_net.ReadOut()[CROSS_DIAGONAL], 0, 0.005)
		)
		return false;
	return true;
}

bool ClamLibTest::CheckReverseNetTraining() const
{
	string path("test");

	string str_forward;
	string str_reverse;

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		path,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);


	// put a square in the ffd network and also in the feedback network
	D_OrientedPattern pat_in = JOCNPattern(SQUARE);
	pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);

	p_tn->_net.ReadIn(pat_in);
	p_tn->_net.Evolve();

	p_rev->ReadIn(p_tn->_net.ReadOut());
	p_rev->Evolve();

	p_rev->CalculateCovariance(p_tn->_net);

	double f_tot = 0;

	for ( SCNodeIterator iter = p_rev->begin(); iter != p_rev->end(); iter++ )
		f_tot += iter->GetValue ();
	
	double cov_sq_sq = f_tot;

	// now put the horizontal cross in the feedback network
	// ovariance between the ffd network and feedback network should be diminished

	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[CROSS_HORIZONTAL] = MAX_LINEAR_STATIC_RATE;

	p_rev->ReadIn(pat_out);
	p_rev->Evolve();
	p_rev->CalculateCovariance(p_tn->_net);

	f_tot = 0;

	for ( SCNodeIterator iter = p_rev->begin(); iter != p_rev->end(); iter++ )
		f_tot += iter->GetValue ();
	
	double cov_sq_hc = f_tot;

	
	// now covariance between diagonal cross and diamond
	pat_in = JOCNPattern(CROSS_DIAGONAL);
	pat_in.ClipMax(MAX_LINEAR_STATIC_RATE);
	p_tn->_net.ReadIn(pat_in);
	p_tn->_net.Evolve();

	pat_out.Clear();
	pat_out[DIAMOND] = MAX_LINEAR_STATIC_RATE;

	p_rev->ReadIn(pat_out);
	p_rev->Evolve();
	p_rev->CalculateCovariance(p_tn->_net);


	f_tot = 0;

	for ( SCNodeIterator iter = p_rev->begin(); iter != p_rev->end(); iter++ )
		f_tot += iter->GetValue ();


	double cov_dc_di = f_tot;


	// finally dc_dc
	pat_out.Clear();
	pat_out[CROSS_DIAGONAL] = MAX_LINEAR_STATIC_RATE;

	p_rev->ReadIn(pat_out);
	p_rev->Evolve();
	p_rev->CalculateCovariance(p_tn->_net);

	f_tot = 0;

	for ( SCNodeIterator iter = p_rev->begin(); iter != p_rev->end(); iter++ )
		f_tot += iter->GetValue ();
	
	double cov_dc_dc = f_tot;

	//equal ojects should lead to a higher covariance
	if (cov_sq_sq  <  2*cov_sq_hc)
		return false;
	if (cov_dc_dc  <  2*cov_dc_di)
		return false;


	return true;
}

bool ClamLibTest::JOCNFWDNetConversionTest() const
{
	string str_forward;
	string str_reverse;

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		TEST_PATH,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);


	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	AddTNToDN convertor;

	D_DynamicNetwork d_net;

	// just take a random input pattern
	D_Pattern pat_in = p_tn->_vec_pattern[12].InPat();

	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&d_net,PERC_OFFSET);
	convertor.Convert
	(
		*p_tn,
		pat_in,
		creator,
		&d_net
	);

	RunTestSimulationSet
	(
		&d_net,
		NAME_JOCN_FORWARD_ROOT,
		ROOT
	);

	return true;
}

bool ClamLibTest::TrainedNetStreamingTest() const
{
	vector<LayerDescription> vec_desc;

	vec_desc.push_back(ClamLib::LAYER_0);
	vec_desc.push_back(ClamLib::LAYER_1);
	vec_desc.push_back(ClamLib::LAYER_2);
	vec_desc.push_back(ClamLib::LAYER_3);
	vec_desc.push_back(ClamLib::LAYER_4);

	vector<D_TrainingUnit> vec_tus = CreateJOCNTrainingUnits();

	SpatialConnectionistNet net_jocn = 
		CreateJOCNFFDNet
		(
			vec_desc,
			vec_tus,
			JOCN_ENERGY
		);

	TrainedNet net(net_jocn,vec_tus);

	ofstream ost_tn("test/jocn.tn");
	if (! ost_tn)
		throw ClamLibException("Couldn't open JOCN test tn file");

	ost_tn << net;

	ost_tn.close();

	ifstream ist_tn("test/jocn.tn");
	if (! ist_tn)
		throw ClamLibException("Couldn't open JOCN test tn file");

	TrainedNet net_in(ist_tn);

	return true;
}

bool ClamLibTest::RunTestSimulationSet
(
	D_DynamicNetwork* p_dnet,
	const string& file_name,
	HandlerMode mode
) const
{
	string absolute_path = TEST_PATH + file_name;

	auto_ptr<AbstractReportHandler> p_handler;

	switch (mode)
	{
	case ASCII:
		p_handler = auto_ptr<AbstractReportHandler>(new AsciiReportHandler(absolute_path));
		break;
	case ROOT:
		p_handler = auto_ptr<AbstractReportHandler>(new RootHighThroughputHandler(absolute_path, true, true));
		break;
	}


	SimulationRunParameter 
		par_run
		(
			*p_handler,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			0.01,       // report time
			0.2,        // update time
			1e-4,       // network step time
			TEST_PATH + file_name + string(".log")
		);

	p_dnet->ConfigureSimulation(par_run);
	p_dnet->Evolve();

	
	return true;
}

namespace {

	Rate functor_test_function(Time time)
	{
		return (time < FUNCTOR_TEST_TIME ) ? 0.0 : MAX_LINEAR_STATIC_RATE;
	}
}

bool ClamLibTest::JOCNFunctorTest() const
{
	string str_forward;
	string str_reverse;

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		TEST_PATH,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	AddTNToDN convertor;

	D_DynamicNetwork d_net;

	// just take a random input pattern
	D_Pattern pat_in = p_tn->_vec_pattern[12].InPat();

	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&d_net,PERC_OFFSET);
	convertor.Convert
	(
		*p_tn,
		pat_in,
		creator,
		&d_net,
		functor_test_function
	);

	RunTestSimulationSet
	(
		&d_net,
		NAME_JOCN_FUNCTOR_ROOT,
		ROOT
	);
	return true;
}

D_DynamicNetwork ClamLibTest::GenerateJocnDynamicNet
(
	const D_Pattern&		pat_ffd,
	const D_Pattern&		pat_rev,
	SimulationOrganizer*	p_organizer
) const
{
	string str_forward;
	string str_reverse;

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		TEST_PATH,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);


	D_DynamicNetwork d_net;

	// The feedback connections need to be pimped up
	p_rev->ScaleWeights(REVERSE_SCALE);


	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&d_net,PERC_OFFSET);
	p_organizer->Convert
	(
		"jocn_dynamic_ffd",
		*p_tn,
		creator,
		pat_ffd,
		&d_net
	);

	// need to reverse the network positions in the reverse network
	// in order to align it with the feedforward network
	p_rev->ReverseAllNetPositions();


	// Have to build a TrainedNet for the feedback network
	TrainedNet tn_back(*p_rev, p_tn->_vec_pattern);


	// displace the feedback network in space
	SpatialPosition pos_offset;
	pos_offset._z = pos_offset._x = pos_offset._f = 0.0F;
	pos_offset._y = VENTRAL_REV_Y_OFFSET;



	p_organizer->Convert
	(
		"jocn_dynamic_rev",
		tn_back,
		creator,
		pat_rev,
		&d_net,
		functor_test_function,
		pos_offset
	);

	return d_net;
}

bool ClamLibTest::JOCNConversionTest() const
{

	// just take a random input pattern

	vector<D_TrainingUnit> vec_tu = CreateJOCNTrainingUnits(); 
	D_Pattern pat_in = vec_tu[12].InPat();

	// generate pattern for the feedback network
	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[CROSS_DIAGONAL] = MAX_LINEAR_STATIC_RATE;

	SimulationOrganizer org;

	D_DynamicNetwork d_net = 
		GenerateJocnDynamicNet
		(
			pat_in,
			pat_out,
			&org
		);

	RunTestSimulationSet
	(
		&d_net,
		NAME_JOCN_FFDFDBCK,
		ROOT
	);

	return true;
}

bool ClamLibTest::JOCNCorrespondenceTest() const
{

	// take four training different patterns at four different positions
	vector<D_TrainingUnit> vec_tu = CreateJOCNTrainingUnits(); 
	D_Pattern pat_in = vec_tu[0].InPat()+  vec_tu[5].InPat() + vec_tu[10].InPat() + vec_tu[15].InPat();

	// generate pattern for the feedback network
	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[CROSS_DIAGONAL] = MAX_LINEAR_STATIC_RATE;

	SimulationOrganizer convertor;

	D_DynamicNetwork d_net = 
		GenerateJocnDynamicNet
		(
			pat_in,
			pat_out,
			&convertor
		);

	// get the original feedforward network to relate their spatial coordinates to their NodeIds
	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;
	string str_forward;
	string str_reverse;

	CreateDevelopingNetworks
	(
		TEST_PATH,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);


	return true;
}

bool ClamLibTest::RootWriteLayerTest() const
{
	// write out a RootLayerDescription
	// The succes must be tested in a ROOT script

	RootLayerDescription desc;
	desc = ToRootLayerDescription(LAYER_1);
	desc.SetName("goeroeboe");

	TFile file("test/rootlayer.root","RECREATE");
	desc.Write();
	file.Close();

	return true;
}

bool ClamLibTest::IdWriteTest() const
{
	TFile file("test/idwritetest.root","RECREATE");
	Id id(42);

	id.Write();
	file.Close();

	return true;
}

bool ClamLibTest::WeightLinkWriteTest() const
{
	TFile file("test/linkwrite.root","RECREATE");

	D_WeightedLink link(Id(99),Id(42),-10.0);
	link.SetName("linkeloetje");
	link.Write();

	file.Close();

	return true;
}

bool ClamLibTest::CircuitInfoReadTest() const
{
	PerceptronCircuitCreator crea;

	CircuitInfo info;
	info.Reserve(crea.Name().c_str(),crea.NumberOfNodes(),Id(7));

	info[PerceptronCircuitCreator::P_OUT] = Id(1);
	info[PerceptronCircuitCreator::N_OUT] = Id(2);

	TFile file("test/circuitinfo.root","RECREATE");
	info.Write();
	file.Close();

	return true;
}

bool ClamLibTest::CircuitInfoWriteTest() const
{
	PerceptronCircuitCreator crea;
	TFile file("test/circuitinfo.root");
	auto_ptr<CircuitInfo> p((CircuitInfo*)file.Get(crea.Name().c_str()));
	cout << p->GetName() << endl;
	
	return true;
}

bool ClamLibTest::RootWriteLayerVectorTest() const
{
	// write out a vector of RootLayerDescription
	// The success  must be tested in a ROOT script

 	if (! TClassTable::GetDict("vector<float>") )
		gROOT->ProcessLine("#include <vector>");

	vector<RootLayerDescription> vec_desc;
	vec_desc.push_back(ToRootLayerDescription(SMALL_LAYER_0));
	vec_desc.push_back(ToRootLayerDescription(SMALL_LAYER_1));

	RootLayeredNetDescription desc(vec_desc);
	TFile file("test/rootlayervector.root","RECREATE");
	desc.SetName("vector_test");
	desc.Write();
	file.Close();

	return true;
}

bool ClamLibTest::SimulationInfoBlockVectorWriteTest() const
{
	if (! TClassTable::GetDict("vector<float>") )
		gROOT->ProcessLine("#include <vector>");
	 
	vector<RootLayerDescription> vec_desc;
	vec_desc.push_back(ToRootLayerDescription(SMALL_LAYER_0));
	vec_desc.push_back(ToRootLayerDescription(SMALL_LAYER_1));

	TFile file("test/simulationinfoblockvector.root","RECREATE");

	vector<CircuitInfo> vec_circ;
	SimulationInfoBlock block("simulationinfoblocktest",vec_desc,vec_circ,CreateSingleDescription());
	vector<SimulationInfoBlock> vec_block;
	vec_block.push_back(block);
	SimulationInfoBlockVector vec(vec_block);
	vec.SetName("vect");
	vec.Write();


	file.Close();

	return true;
}

bool ClamLibTest::SimulationInfoBlockVectorReadTest() const
{
 	gROOT->ProcessLine("#include <vector>");
	TFile file("test/simulationinfoblockvector.root");
	
	SimulationInfoBlockVector* p_vec = (SimulationInfoBlockVector*)file.Get("vect");

	vector<SimulationInfoBlock> vec = p_vec->BlockVector();
	cout << vec[0].DescriptionVector()[0]._nr_x_pixels << endl;
	cout << vec[0].DescriptionCircuit().GetName() << endl;
	return true;
}

bool ClamLibTest::CircuitCreatorProxyTest() const
{

	return true;
}

namespace {

	DynamicLib::Rate SmallNetMetaInput(DynamicLib::Time time)
	{
		return (time < 0.5) ? 0.0 : 1.0;
	}
}

TrainedNet ClamLibTest::GenerateSmallDirectMetaNet() const
{
	// Generate a small ffd net with a positive squashing function
	// to check if the correct relations between DynamicNet NodeIds and
	// ffd net NodeIds are generated correctly and stored with the DynamicNetwork

	// first generate a small network
	vector<LayerDescription> vec_desc;
	vec_desc.push_back(ClamLib::SMALL_LAYER_0);
	vec_desc.push_back(ClamLib::SMALL_LAYER_1);
	vec_desc.push_back(ClamLib::SMALL_LAYER_2);

	DenseOverlapLinkRelation rel(vec_desc);
	
	// by default a network with an anti-symmetric squashing function is created, 
	// which we don't want here
	SigmoidParameter par_squash(0,1,1);
	Sigmoid sig(par_squash);
	SpatialConnectionistNet net(&rel,sig);


	// also generate an input pattern and an output pattern so that we can generate a TrainedNet
	// do not bother about training though

	D_OrientedPattern pat_in(SMALL_LAYER_0._nr_x_pixels, SMALL_LAYER_0._nr_y_pixels, SMALL_LAYER_0._nr_features);

	// can I read it in?
	net.ReadIn(pat_in);

	// still alive so yes
	D_Pattern pat_out(SMALL_LAYER_2._nr_x_pixels*SMALL_LAYER_2._nr_y_pixels*SMALL_LAYER_2._nr_features);

	pat_out = net.ReadOut();

	D_TrainingUnit tu(pat_in,pat_out);
	vector<D_TrainingUnit> vec_tu;
	vec_tu.push_back(tu);

	TrainedNet tn(net,vec_tu);

	return tn;
}

bool ClamLibTest::SmallNetMetaTest() const
{
	// Purpose: to test whether there is a correspondence between the NodeId values of the ANN and the NodeId values
	// of the DynamicNetwork
	TrainedNet tn = this->GenerateSmallDirectMetaNet();
	D_Pattern pat_in = tn._vec_pattern[0].InPat();

	// now the DynamicNetwork where it is all going to happen
	D_DynamicNetwork net_dynamic;

	D_OrientedPattern 
		pat_dynamic_input
		(	
			SMALL_LAYER_0._nr_x_pixels, 
			SMALL_LAYER_0._nr_y_pixels, 
			SMALL_LAYER_0._nr_features
		);

	pat_dynamic_input.Clear();
	pat_dynamic_input(0,0,0) = 1.0;
	pat_dynamic_input(0,1,0) = 1.0;
	pat_dynamic_input(1,1,1) = 1.0;

	// do the conversion
	AddTNToDN convertor;
	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);


	SimpleCircuitCreator creator(&sig_exc,&sig_inh,&net_dynamic,PERC_OFFSET);
	convertor.Convert
	(
		tn,
		pat_dynamic_input,
		creator,
		&net_dynamic,
		SmallNetMetaInput
	);

	// there are 2*2*2 + 2*2 + 2 + 8 (input) = 22 nodes in the network
	if (  net_dynamic.NumberOfNodes() != 22 )
		return false;

	return true;
}

bool ClamLibTest::SimulationInfoBlockWriteTest() const
{
	// purpose is to write a SimulationOrganizer object into the simulation
	// root file and to be able to match 

 	gROOT->ProcessLine("#include <vector>");

	TrainedNet tn = this->GenerateSmallDirectMetaNet();
	D_Pattern pat_in = tn._vec_pattern[0].InPat();

	// now the DynamicNetwork where it is all going to happen
	D_DynamicNetwork net_dynamic;

	D_OrientedPattern pat_dynamic_input(SMALL_LAYER_0._nr_x_pixels, SMALL_LAYER_0._nr_y_pixels, SMALL_LAYER_0._nr_features);

	pat_dynamic_input.Clear();
	pat_dynamic_input(0,0,0) = 1.0;
	pat_dynamic_input(1,1,1) = 1.0;


	SimulationOrganizer org;

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&net_dynamic,PERC_OFFSET);

	NetId id = 
		org.Convert
		(
			"small_net",
			tn,
			creator,
			pat_dynamic_input,
			&net_dynamic,
			SmallNetMetaInput
		);

	// write out the dynamic network
	ofstream str_dynamic_meta("test/dynamic_meta.net");
	net_dynamic.ToStream(str_dynamic_meta);

	// write the single SimulationInfo block info into a root file to analyze it
	TFile file("test/smallmetatestinfoblock.root","RECREATE");
	org[0].Write();
	file.Close();

	return true;
}

bool ClamLibTest::SimulationInfoBlockReadTest() const
{
	TFile file("test/smallmetatestinfoblock.root");
	auto_ptr<SimulationInfoBlock> p_block((SimulationInfoBlock*)file.Get("small_net"));	
	file.Close();

	vector<CircuitInfo> vec_info= p_block->InfoVector();
	cout << vec_info.size() << endl;

	CircuitInfo inf = vec_info[1];
	cout << inf.GetName() << endl;

	return true;
}

bool ClamLibTest::CircuitFactoryTest() const
{
	// deprecated

	return true;
}

bool ClamLibTest::SmallPositiveInfoTest() const 
{
	// Purpose: to write a SimulationInfoBlock into the same file as the simulation results.
	//			The simulation is a replication (code replication ...) 
	//			of the smallpositive simulation above, i.e. a small
	//			network that uses direct circuit conversion. The usage is atypical because in the
	//			long run the SimulationOrganizer will probably be used for both the conversions
	//			and running the simulation, but at this moment this not implemented.

	// Author:	Marc de Kamps
	// Date:	18-03-2009

	TrainedNet tn = this->GenerateSmallNet(DIRECT);
	ofstream str_net("test/smallpositiveinfo11.net");
	if (! str_net)
		return false;
	str_net << tn._net;

	D_Pattern in(2);
	in[0] = 1.0;
	in[1] = 1.0;

	D_Pattern dummy(2);
	tn._vec_pattern[0] = D_TrainingUnit(in,dummy);

	D_DynamicNetwork d_net11;

	auto_ptr<SimulationInfoBlock> p_block(new SimulationInfoBlock);
	this->OrganizerConversionTest
	(
		"smallpositiveinfo11",
		DIRECT, 
		tn, 
		&d_net11,
		p_block
	);

	string str_absname11("test/smallpositiveinfo11.root");
	RootReportHandler handler11(str_absname11);

	SimulationRunParameter 
		par_run
		(
			handler11,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/smallpositiveinfo11.txt"
		);

	d_net11.ConfigureSimulation(par_run);
	d_net11.Evolve();

	TFile file_11("test/smallpositiveinfo11.root","UPDATE");
	p_block->SetName("smallpositiveinfo11");
	p_block->Write();
	file_11.Close();

    // now the same for 00
	in[0] = 0;
	in[1] = 0;

	tn._vec_pattern[0] = D_TrainingUnit(in,dummy);

	D_DynamicNetwork d_net00;
	this->OrganizerConversionTest
	(
		"smallpositiveinfo00",
		DIRECT, 
		tn, 
		&d_net00,
		p_block
	);

	string str_absname00("test/smallpositiveinfo00.root");
	RootReportHandler handler00(str_absname00);

	SimulationRunParameter 
		par_run00
		(
			handler00,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/smallpositiveinfo00.txt"
		);

	d_net00.ConfigureSimulation(par_run00);
	d_net00.Evolve();

	TFile file_00("test/smallpositiveinfo00.root","UPDATE");
	p_block->SetName("smallpositiveinfo00");
	p_block->Write();
	file_00.Close();

	return true;
}
bool ClamLibTest::OrganizerConversionTest
(
	const string&					name,
	const ConversionMode&			mode, 
	const TrainedNet&				network, 
	D_DynamicNetwork*				p_dnet,
	auto_ptr<SimulationInfoBlock>&	p_block
) const
{
	// Purpose: Generate a DynamicNetwork from a trained SparseLayeredNetwork.
	//			The type of network is determined by the mode. It is closely
	//			related to GenerateConversionTest, but uses a SimulationOrganizer,
	//			rather than a AddTNToDN so that a SimulationInfoBlock associated
	//			with the conversion is created
	//
	// Author:  Marc de Kamps
	// Date:    17-03-2009


	// create an input field so that the initial conditions of the network can be set.
	// the rate functions of the network will be read from this pattern
	OrientedPattern<DynamicLib::Rate> 
		input_field
		(
			network._net.Dimensions()[0]._nr_x_pixels,
			network._net.Dimensions()[0]._nr_y_pixels,
			network._net.Dimensions()[0]._nr_features
		);

	D_Pattern pat_trained = network._vec_pattern[0].InPat();
    SpatialConnectionistNet net = network._net;
	for ( Index i = 0; i < pat_trained.Size(); i++ )
		input_field[i] = pat_trained[i];


	if (mode == DIRECT)
		p_dnet->SetDalesLaw(false);

	WilsonCowanParameter param_exc(20e-3,1,1);
	WilsonCowanParameter param_inh(10e-3,1,1);

	auto_ptr<D_AbstractAlgorithm> p_exc;
	auto_ptr<D_AbstractAlgorithm> p_inh;
	
	if ( mode == DIRECT)
	{
		p_exc = auto_ptr<WilsonCowanAlgorithm>(new WilsonCowanAlgorithm(param_exc));
		p_inh = auto_ptr<WilsonCowanAlgorithm>(new WilsonCowanAlgorithm(param_inh));
	}
	else
	{
		p_exc = auto_ptr<SemiSigmoid>(new SemiSigmoid(param_exc));
		p_inh = auto_ptr<SemiSigmoid>(new SemiSigmoid(param_inh));
	}

	PerceptronCircuitCreator creator(p_exc.get(),p_inh.get(),p_dnet,PERC_OFFSET);
	SimulationOrganizer convert;

	convert.Convert
	(
		name,
		network,
		creator,
		input_field,
		p_dnet
	);

	
	return true;
}

bool ClamLibTest::SimulationOrganizerSmallDirectTest() const
{
	// Purpose: To run the same simulation as in SmallPositiveInfoTest, but now properly, as would be done
	//			by the typical user. In particular the SimulationInfoBlock objects are no longer relevant 
	//			for the client. The SimulationOrganizer now has to confirm to the ClamStorageFormat.
	// Author:	Marc de Kamps
	// Date:	18-03-2009

 	gROOT->ProcessLine("#include <vector>");

	TrainedNet tn = this->GenerateSmallNet(DIRECT);
	ofstream str_net("test/simulationorganizersmalldirect.net");
	if (! str_net)
		return false;
	str_net << tn._net;

	D_Pattern in(2);
	in[0] = 1.0;
	in[1] = 1.0;

	D_Pattern dummy(2);
	tn._vec_pattern[0] = D_TrainingUnit(in,dummy);

	D_DynamicNetwork d_net11;
	d_net11.SetDalesLaw(false);

	string str_absname11("test/simulationorganizersmalldirect.root");
	RootReportHandler handler11(str_absname11);

	SimulationRunParameter 
		par_run
		(
			handler11,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/simulationorganizersmalldirect.txt"
		);

	WilsonCowanParameter param_exc(20e-3,1,1);
	WilsonCowanAlgorithm alg(param_exc);

	SimpleCircuitCreator creator(&alg,0,&d_net11,NO_OFFSET);
	SimulationOrganizer org;
	org.Convert
	(
		"simulationorganizersmalldirect",
		tn,
		creator,
		in,
		&d_net11
	);

	org.Configure(par_run);

	org.Evolve();

	ofstream str_dynamic("test/simulationorganizersmalldirect.dynnet");
	str_dynamic << d_net11;

	return true;
}

bool ClamLibTest::GenericJocnTest
(
	const string&					simulation_name,
	D_DynamicNetwork*				p_dnet,		
	const AbstractCircuitCreator&	creator,
	SimulationOrganizer*			p_org,
	bool							ffd
 ) const
{
	if (! TClassTable::GetDict("vector<float>") )
		gROOT->ProcessLine("#include <vector>");

	// take four training different patterns at four different positions
	vector<D_TrainingUnit> vec_tu = CreateJOCNTrainingUnits(); 
	D_Pattern pat_in = vec_tu[0].InPat()+  vec_tu[5].InPat() + vec_tu[10].InPat() + vec_tu[15].InPat();

	// generate pattern for the feedback network
	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[SQUARE] = MAX_LINEAR_STATIC_RATE;

	string str_forward;
	string str_reverse;

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		TEST_PATH,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	
	ostringstream ost;
	ost << "test/" << simulation_name << ".root";
	string str_jocnffd(ost.str());
	RootReportHandler handlerjocnffd(str_jocnffd);
	SimulationRunParameter 
		par_run
		(
			handlerjocnffd,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			simulation_name
		);

	bool b_extern = true;
	if (! p_org)
	{
		p_org = new SimulationOrganizer;
		b_extern = false;
	}

	p_org->Convert
	(
		simulation_name,
		*p_tn,
		creator,
		pat_in,
		p_dnet
	);
	if (! ffd) {
		D_TrainingUnit dummy(pat_out,pat_out);
		vector<D_TrainingUnit> vec_tuout;
		vec_tu.push_back(dummy);

		TrainedNet rev(*p_rev,vec_tuout);
		HomeostaticSmooth(&rev,HOMEOSTATIC_SMOOTH_SCALE);

		// Convert the feedback network
		p_org->Convert
		(
			"simulationjocnfbk",
			rev,
			creator,
			pat_out,
			p_dnet,
			FeedbackRate
		);
	}


	if (! b_extern){
		p_org->Configure(par_run);
		p_org->Evolve();
		delete p_org;
	}

	return true;
}

bool ClamLibTest::SimulationInfoJOCNFFDTest() const
{
	D_DynamicNetwork d_netjocnffd;

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&d_netjocnffd,PERC_OFFSET);

	bool b_ret = this->GenericJocnTest("simulationjocnffd",&d_netjocnffd, creator);

	ofstream str_net("test/ffd.net");
	str_net << d_netjocnffd;

	return b_ret;
}

bool ClamLibTest::SimulationInfoJOCNTest() const
{
	if (! TClassTable::GetDict("vector<float>") )
		gROOT->ProcessLine("#include <vector>");

	// take four fifferent ojects at four different positions
	vector<D_TrainingUnit> vec_tu = CreateJOCNTrainingUnits(); 
	D_Pattern pat_in = vec_tu[0].InPat();
	pat_in += vec_tu[5].InPat();
	pat_in += vec_tu[10].InPat();
	pat_in += vec_tu[15].InPat();

	// generate pattern for the feedback network
	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[CROSS_DIAGONAL] = MAX_LINEAR_STATIC_RATE;

	string str_forward;
	string str_reverse;

	TrainedNet* p_tn = 0;
	SpatialConnectionistNet* p_rev = 0;

	CreateDevelopingNetworks
	(
		TEST_PATH,
		&str_forward,
		&str_reverse,
		&p_tn,
		&p_rev
	);

	string str_jocnffd("test/simulationjocn.root");
	RootReportHandler handlerjocnffd(str_jocnffd);
	SimulationRunParameter 
		par_run
		(
			handlerjocnffd,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/simulationjocn.txt"
		);

	D_DynamicNetwork d_netjocn;

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&d_netjocn,PERC_OFFSET);
	// Convert the feedforward network
	SimulationOrganizer org;
	org.Convert
	(
		"simulationjocnffd",
		*p_tn,
		creator,
		pat_in,
		&d_netjocn
	);

	D_TrainingUnit dummy(pat_out,pat_out);
	vector<D_TrainingUnit> vec_tuout;
	vec_tu.push_back(dummy);

	TrainedNet rev(*p_rev,vec_tuout);

	// Convert the feedback network
	org.Convert
	(
		"simulationjocnfbk",
		rev,
		creator,
		pat_out,
		&d_netjocn,
		FeedbackRate
	);
	org.Configure(par_run);

	org.Evolve();
	return true;
}

string ClamLibTest::GenerateSmallSimulationFile() const
{
	if (! TClassTable::GetDict("vector<float>") )
		gROOT->ProcessLine("#include <vector>");

	// generate a small network

	vector<LayerDescription> desc;
	desc.push_back(SMALL_LAYER_0);
	desc.push_back(SMALL_LAYER_1);
	desc.push_back(SMALL_LAYER_2);

	DenseOverlapLinkRelation rel(desc);
	SpatialConnectionistNet net(&rel);

	// and an arbitraty input pattern
	D_Pattern pat_in(net.NumberOfInputNodes());
	D_Pattern pat_out(net.NumberOfOutputNodes());
	vector<D_TrainingUnit> vec_pat;
	vec_pat.push_back(D_TrainingUnit(pat_in,pat_out));

	// to create a fullgrwn TrainedNet
	TrainedNet tn(net,vec_pat);

	D_DynamicNetwork d_net;
	
	string str_itertest("test/iteratortest.root");
	RootReportHandler handlerjocnffd(str_itertest);
	SimulationRunParameter 
		par_run
		(
			handlerjocnffd,
			10000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"test/iterator.txt"
		);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronCircuitCreator creator(&sig_exc,&sig_inh,&d_net,PERC_OFFSET);

	// Convert the feedforward network
	SimulationOrganizer org;
	org.Convert
	(
		"iteratortest",
		tn,
		creator,
		pat_in,
		&d_net
	);

	org.Configure(par_run);
	org.Evolve();

	return str_itertest;
}

bool ClamLibTest::SubNetworkIteratorTest() const
{
	string simulation_file = this->GenerateSmallSimulationFile();
	TFile file(simulation_file.c_str());
	SimulationResult sim(file);

	// Get out the first DynamicSubNetwork;
	DynamicSubNetwork sub_net = *sim.begin();

	DynamicSubNetwork::const_iterator iter = sub_net.begin();

	for
	(
		DynamicSubNetwork::const_iterator iter = sub_net.begin();
		iter != sub_net.end();
		iter++
	)
	{
		PhysicalPosition pos;
		if (iter->NumberOfNodes() > 0 ){
			iter.Position(&pos);
			cout << iter->IdOriginal()		<< " ";
			cout << pos._position_x			<< " ";
			cout << pos._position_y			<< " ";
			cout << pos._position_z			<< " ";
			cout << pos._position_depth		<< endl;
		}
	}
	return true;
}

bool ClamLibTest::ReverseSubNetworkIteratorTest() const
{	string simulation_file = this->GenerateSmallSimulationFile();
	TFile file(simulation_file.c_str());
	SimulationResult sim(file);

	cout << "---" << endl;
	// Get out the first DynamicSubNetwork;
	DynamicSubNetwork sub_net = *sim.begin();

	for
	(
		DynamicSubNetwork::const_rziterator iter = sub_net.rzbegin();
		iter != sub_net.rzend();
		iter++
	)
	{
		PhysicalPosition pos;
		if (iter->NumberOfNodes() > 0 ){
			iter.Position(&pos);
			cout << iter->IdOriginal()		<< " ";
			cout << pos._position_x			<< " ";
			cout << pos._position_y			<< " ";
			cout << pos._position_z			<< " ";
			cout << pos._position_depth		<< endl;
		}
	}

	return true;
}

bool ClamLibTest::SimulationResultIteratorTest() const
{
	string simulation_file = this->GenerateSmallSimulationFile();

	TFile file(simulation_file.c_str());
	SimulationResult sim(file);
	
	for (SimulationResult::iterator iter = sim.begin(); iter != sim.end(); iter++ )
		if ( iter->GetName() != string("iteratortest") )
			return false;

	return true;
}

bool ClamLibTest::CircuitNodeRoleSerializationTest() const
{
	if (! TClassTable::GetDict("vector<float>") )
		gROOT->ProcessLine("#include <vector>");
	CircuitNodeRole role_simple_p("p",DynamicLib::EXCITATORY,0.1F,-0.1F,0.5F,0.0F);
	CircuitNodeRole role_simple_n("n",DynamicLib::EXCITATORY,1.0F,1.0F,1.0F,1.0F);
	IndexWeight iw;
	iw._name_predecessor = "p";
	iw._weight =-5.0;
	bool b_ret = role_simple_n.AddIncoming(iw);

	if (! b_ret )
		return false;

	string file_name("test/roletest.root");
	TFile file(file_name.c_str(),"RECREATE");
	role_simple_n.Write();
	file.Close();

	TFile file_role_and_rock("test/roletest.root");
	file_role_and_rock.ls();

	CircuitDescription desc(2);
	desc.SetName("kadaver");
	desc.AddExternal("p");
	desc.push_back(role_simple_p);
	desc.push_back(role_simple_n);
	

	TFile file_desc("test/description.root","RECREATE");
	desc.Write();
	file_desc.Close();

	TFile file_read("test/description.root");
	CircuitDescription* p_desc = (CircuitDescription*)file_read.Get("kadaver");
	if (! p_desc)
		return false;

	return true;
}

bool ClamLibTest::IndexWeightSerializationTest() const
{
	IndexWeight ind;
	ind._name_predecessor = "bla";
	ind._weight = -100;
	ind._index = 10;

	TFile file("test/indexweight.root","RECREATE");
	ind.Write();
	file.Close();

	return true;
}

bool ClamLibTest::ConfigurableCreatorTest() const
{
	// emulate the functionality of the SimpleCircuitCreator
	CircuitNodeRole role("the_node",DynamicLib::EXCITATORY,0.0,0.0,0.0,0.0);

	CircuitDescription desc(1);
	desc.AddExternal("the_node"); // now necessary, add before finalizing the node
	Index id_single = desc.push_back(role);

	InputOutputPair pair;
	pair._id_in  = id_single;
	pair._id_out = id_single;
	desc.push_back_io(pair);
	desc.SetName("simple");

	D_DynamicNetwork net;
	net.SetDalesLaw(false);
	TrainedNet tn = this->GenerateSmallNet(DIRECT);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	ConfigurableCreator creator(&sig_exc,&sig_inh,&net,desc);

	AddTNToDN convert;

	D_Pattern pat(tn._net.NumberOfInputNodes());

	convert.Convert(tn,pat,creator,&net);

	ofstream stream("simple_new.net");
	stream << net;
	return true;
}

bool ClamLibTest::PerceptronConfigurableCreatorTest() const
{
	// emulate the functionality of the SimpleCircuitCreator

	CircuitDescription desc(6);

	IndexWeight iw;

	CircuitNodeRole p_out("P_OUT",DynamicLib::EXCITATORY, 1.0,0.0,1.0,0.0);
	CircuitNodeRole n_out("N_OUT",DynamicLib::EXCITATORY,-1.0,0.0,1.0,0.0);
	CircuitNodeRole e_p  ("e_p",  DynamicLib::EXCITATORY, 1.5,0.0,0.0,0.0);
	CircuitNodeRole e_n  ("e_n",  DynamicLib::EXCITATORY,-1.5,0.0,0.0,0.0);
	CircuitNodeRole i_n  ("i_n",  DynamicLib::INHIBITORY, 1.5,0.0,0.0,0.0);
	CircuitNodeRole i_p  ("i_p",  DynamicLib::INHIBITORY, 0.5,0.0,0.0,0.0);

	double circuit_weight = 2.0;
	p_out.AddIncoming(IndexWeight("e_p",circuit_weight));
	n_out.AddIncoming(IndexWeight("i_p",-circuit_weight));
	n_out.AddIncoming(IndexWeight("e_n",circuit_weight));
	p_out.AddIncoming(IndexWeight("i_n",-circuit_weight));

	desc.AddExternal("e_p");

	desc.push_back(e_p);
	desc.push_back(e_n);
	desc.push_back(i_p);
	desc.push_back(i_n);
	desc.push_back(p_out);
	desc.push_back(n_out);

	//ignore the io-pairs because they will be handed by the overloaded perceptroncreator function anyway

	D_DynamicNetwork net;

	TrainedNet tn = this->GenerateSmallNet(DIRECT);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronConfigurableCreator creator(&sig_exc,&sig_inh,&net,desc);

	AddTNToDN convert;

	D_Pattern pat(tn._net.NumberOfInputNodes());

	convert.Convert(tn,pat,creator,&net);

	ofstream stream("perc_new.net");
	stream << net;
	return true;
}

bool ClamLibTest::SimulationInfoJOCNFFDConfigurableTest() const
{
	D_DynamicNetwork d_netjocnffd;

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	CircuitDescription desc(6);

	IndexWeight iw;

	CircuitNodeRole p_out("P_OUT",DynamicLib::EXCITATORY, 1.0,0.0,1.0,0.0);
	CircuitNodeRole n_out("N_OUT",DynamicLib::EXCITATORY,-1.0,0.0,1.0,0.0);
	CircuitNodeRole e_p  ("e_p",  DynamicLib::EXCITATORY, 1.5,0.0,0.0,0.0);
	CircuitNodeRole e_n  ("e_n",  DynamicLib::EXCITATORY,-1.5,0.0,0.0,0.0);
	CircuitNodeRole i_n  ("i_n",  DynamicLib::INHIBITORY, 1.5,0.0,0.0,0.0);
	CircuitNodeRole i_p  ("i_p",  DynamicLib::INHIBITORY, 0.5,0.0,0.0,0.0);

	double circuit_weight = 2.0;
	p_out.AddIncoming(IndexWeight("e_p",circuit_weight));
	n_out.AddIncoming(IndexWeight("i_p",-circuit_weight));
	n_out.AddIncoming(IndexWeight("e_n",circuit_weight));
	p_out.AddIncoming(IndexWeight("i_n",-circuit_weight));

	desc.AddExternal("e_p");

	desc.push_back(e_p);
	desc.push_back(e_n);
	desc.push_back(i_p);
	desc.push_back(i_n);
	desc.push_back(p_out);
	desc.push_back(n_out);

	PerceptronConfigurableCreator creator(&sig_exc,&sig_inh,&d_netjocnffd,desc);

	bool b_ret =  this->GenericJocnTest("simulationjocnconfigurableffd",&d_netjocnffd, creator);

	return b_ret;
}

bool ClamLibTest::TestStockDescriptions() const
{
	cout << CreateSingleDescription().GetName() << endl;
	cout << CreatePerceptronDescription().GetName();
	return true;
}

bool ClamLibTest::DynamicSubLayeredIteratorTest() const
{
	D_DynamicNetwork net;

	TrainedNet tn = this->GenerateSmallNet(CIRCUIT);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronConfigurableCreator creator(&sig_exc,&sig_inh,&net,CreatePerceptronDescription());

	SimulationOrganizer convert;

	D_Pattern pat(tn._net.NumberOfInputNodes());
	pat.Clear();
	pat[0] = 0.01;
	pat[1] = 0.01;

	convert.Convert("subiter",tn,creator,pat,&net);

	ofstream stream("subiter.net");
	stream << net;
	
	for 
	(
		DynamicSubNetwork::const_iterator iter = convert[0].begin(); 
		iter != convert[0].end(); 
		iter++
	)
	{
		PhysicalPosition pos;
		if (iter->NumberOfNodes() > 0 ){
			iter.Position(&pos);
			cout << iter->IdOriginal()		<< " ";
			cout << pos._position_x			<< " ";
			cout << pos._position_y			<< " ";
			cout << pos._position_z			<< " ";
			cout << pos._position_depth		<< endl;
		}
	}

	for 
	(
		DynamicSubNetwork::const_iterator iter_layer_0 = convert[0].begin(0); 
		iter_layer_0 != convert[0].end(0); 
		iter_layer_0++
	)
	{
		PhysicalPosition pos;
		if (iter_layer_0->NumberOfNodes() > 0 ){
			iter_layer_0.Position(&pos);
			cout << iter_layer_0->IdOriginal()		<< " ";
			cout << pos._position_x			<< " ";
			cout << pos._position_y			<< " ";
			cout << pos._position_z			<< " ";
			cout << pos._position_depth		<< endl;
		}		
	}
	for 
	(
		DynamicSubNetwork::const_iterator iter_layer_1 = convert[0].begin(1); 
		iter_layer_1 != convert[0].end(1); 
		iter_layer_1++
	)
	{
		PhysicalPosition pos;
		if (iter_layer_1->NumberOfNodes() > 0 ){
			iter_layer_1.Position(&pos);
			cout << iter_layer_1->IdOriginal()		<< " ";
			cout << pos._position_x					<< " ";
			cout << pos._position_y					<< " ";
			cout << pos._position_z					<< " ";
			cout << pos._position_depth				<< endl;
		}		
	}

	// this is legal, because it is equivalent to end()
	DynamicSubNetwork::const_iterator iter_layer_2 = convert[0].begin(2); 

	// but an attempt to dereference it is not
	try {
		Number n = iter_layer_2->NumberOfNodes();

		// do something nonsensical to shut up compiler warnings; the code should never get her since by this time an exception should be thrown
		n+=2;
	}
	catch(ClamLibException& exc)
	{
		cout << "caught exception, as expected: " << exc.Description() << endl;
	}

	// this is not legal, there is no useful interpretation
	try {
		DynamicSubNetwork::const_iterator iter_layer_22 = convert[0].end(2); 
	}
	catch(ClamLibException& exc)
	{
		cout << "caught exception, as expected: " << exc.Description() << endl;
	}
	return true;
}

bool ClamLibTest::DynamicSubLayeredReverseIteratorTest() const
{
	D_DynamicNetwork net;

	TrainedNet tn = this->GenerateSmallNet(CIRCUIT);

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	PerceptronConfigurableCreator creator(&sig_exc,&sig_inh,&net,CreatePerceptronDescription());

	SimulationOrganizer convert;

	D_Pattern pat(tn._net.NumberOfInputNodes());
	pat.Clear();
	pat[0] = 0.01;
	pat[1] = 0.01;

	convert.Convert("subiterrz",tn,creator,pat,&net);

	ofstream stream("subiterrz.net");
	stream << net;
	
	for 
	(
		DynamicSubNetwork::const_rziterator iter = convert[0].rzbegin(); 
		iter != convert[0].rzend(); 
		iter++
	)
	{
		PhysicalPosition pos;
		if (iter->NumberOfNodes() > 0 ){
			iter.Position(&pos);
			cout << iter->IdOriginal()		<< " ";
			cout << pos._position_x			<< " ";
			cout << pos._position_y			<< " ";
			cout << pos._position_z			<< " ";
			cout << pos._position_depth		<< endl;
		}
	}	


	for 
	(
		DynamicSubNetwork::const_rziterator iter_layer_0 = convert[0].rzbegin(0); 
		iter_layer_0 != convert[0].rzend(0); 
		iter_layer_0++
	)
	{
		PhysicalPosition pos;
		if (iter_layer_0->NumberOfNodes() > 0 ){
			iter_layer_0.Position(&pos);
			cout << iter_layer_0->IdOriginal()		<< " ";
			cout << pos._position_x					<< " ";
			cout << pos._position_y					<< " ";
			cout << pos._position_z					<< " ";
			cout << pos._position_depth				<< endl;
		}		
	}

	for 
	(
		DynamicSubNetwork::const_rziterator iter_layer_1 = convert[0].rzbegin(1); 
		iter_layer_1 != convert[0].rzend(1); 
		iter_layer_1++
	)
	{
		PhysicalPosition pos;
		if (iter_layer_1->NumberOfNodes() > 0 ){
			iter_layer_1.Position(&pos);
			cout << iter_layer_1->IdOriginal()		<< " ";
			cout << pos._position_x					<< " ";
			cout << pos._position_y					<< " ";
			cout << pos._position_z					<< " ";
			cout << pos._position_depth				<< endl;
		}		
	}

	// this is legal, because it is equivalent to end()
	DynamicSubNetwork::const_rziterator iter_layer_2r = convert[0].rzbegin(2); 

	// but an attempt to dereference it is not
	try {
		Number n = iter_layer_2r->NumberOfNodes();

		// do something nonsensical to shut up compiler warnings about variable not being used
		n++;
	}
	catch(ClamLibException& exc)
	{
		cout << "caught exception, as expected: " << exc.Description() << endl;
	}

	// this is not legal, there is no useful interpretation
	try {
		DynamicSubNetwork::const_iterator iter_layer_2r = convert[0].end(2); 
	}
	catch(ClamLibException& exc)
	{
		cout << "caught exception, as expected: " << exc.Description() << endl;
	}
	return true;
}

bool ClamLibTest::JOCNIteratorTest() const
{

	// just take a random input pattern

	vector<D_TrainingUnit> vec_tu = CreateJOCNTrainingUnits(); 
	D_Pattern pat_in = vec_tu[12].InPat();

	// generate pattern for the feedback network
	D_Pattern pat_out(4);
	pat_out.Clear();
	pat_out[CROSS_DIAGONAL] = MAX_LINEAR_STATIC_RATE;

	SimulationOrganizer org;

	D_DynamicNetwork d_net = 
		GenerateJocnDynamicNet
		(
			pat_in,
			pat_out,
			&org
		);

	// Let's get the V4 layer of the feedfoward and the feedback network, we should
	// be able to do this by iterator

	const int V2=0;

	const int REV = 1;


	for 
	( 
		DynamicSubNetwork::const_rziterator iter = org[REV].rzbegin(V2);
		iter != org[REV].rzend(V2);
		iter++
	)
	{
		PhysicalPosition pos_f;
		iter.Position(&pos_f);
		cout << iter->IdOriginal()		    << " ";
		cout << pos_f._position_x			<< " ";
		cout << pos_f._position_y			<< " ";
		cout << pos_f._position_z			<< " ";
		cout << pos_f._position_depth		<< endl;	

	}

	return true;
}

bool ClamLibTest::SimulationInfoJOCNConfigurableTest() const
{
	D_DynamicNetwork d_netjocnffd;

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);


	PerceptronConfigurableCreator creator(&sig_exc,&sig_inh,&d_netjocnffd,CreatePerceptronDescription());
	bool b_ret =  this->GenericJocnTest("simulationjocnconfigurable",&d_netjocnffd, creator,0,false);
	// simulate full network                                                                   ^

	return b_ret;	return true;
}

bool ClamLibTest::DisInhibitionTest() const
{
	D_DynamicNetwork d_netjocn;

	SemiSigmoid sig_exc(PARAM_EXC);
	SemiSigmoid sig_inh(PARAM_INH);

	SimulationOrganizer org;
	PerceptronConfigurableCreator creator(&sig_exc,&sig_inh,&d_netjocn,CreatePerceptronDescription());
	bool b_ret =  this->GenericJocnTest("disinhibitionffd",&d_netjocn, creator,&org,false);
	if (! b_ret )
		return false;

	// simulate full network                                                                      ^

	// First create we need to create  a TrainedNet that we can convert in to a DynamicSubNetwork
	// It will be a three layered network that will be between the V2, V4 and PIT layers of the feedforward and feedback network

	vector<LayerDescription> vec_dis;
	const LayerDescription NULL_LAYER =
	{
		0,  // nr x pixels
		0,  // nr y pixels
		0,  // nr orientations
		0,  // size of receptive field in x
		0,  // size of receptive field in y
		0,  // nr x skips
		0   // nr y skips
	};
	vec_dis.push_back(NULL_LAYER);
	vec_dis.push_back(ClamLib::LAYER_1);
	vec_dis.push_back(ClamLib::LAYER_2);
	vec_dis.push_back(ClamLib::LAYER_3);

	NoLinkRelation link_rel(vec_dis);
	SpatialConnectionistNet net(&link_rel);

	vector<D_TrainingUnit> vec_dummy;

	TrainedNet tn(net,vec_dummy);

	CircuitDescription desc(6);

	static const bool OUTPUT = true;
	static const bool POSITIVE = true;
	static const bool NEGATIVE = false;
	CircuitNodeRole e_dis_p("e_dis_p",DynamicLib::EXCITATORY, 1.0,0.0,1.0,0.0, OUTPUT, POSITIVE);
	CircuitNodeRole i_dis_p("i_dis_p",DynamicLib::INHIBITORY,-1.0,0.0,-1.0,0.0);
	CircuitNodeRole i_gat_p("i_gat_p",DynamicLib::INHIBITORY, 1.5,0.0,0.0,0.0);

	CircuitNodeRole e_dis_n("e_dis_n",DynamicLib::EXCITATORY, -1.0,0.0,1.0,0.0, OUTPUT, NEGATIVE);
	CircuitNodeRole i_dis_n("i_dis_n",DynamicLib::INHIBITORY,-1.0,0.0,-1.0,0.0);
	CircuitNodeRole i_gat_n("i_gat_n",DynamicLib::INHIBITORY, -1.5,0.0,0.0,0.0);


	const double w_iip = -1.0;
	e_dis_p.AddIncoming(IndexWeight("i_gat_p",w_iip));
	i_gat_p.AddIncoming(IndexWeight("i_dis_p",w_iip));

	e_dis_n.AddIncoming(IndexWeight("i_gat_n",w_iip));
	i_gat_n.AddIncoming(IndexWeight("i_dis_n",w_iip));

	desc.AddExternal("e_dis_p"); //TODO: examine if the externals are always necessary
	desc.AddExternal("e_dis_n");

	desc.push_back(e_dis_p);
	desc.push_back(i_dis_p);	
	desc.push_back(i_gat_p);	
	desc.push_back(e_dis_n);
	desc.push_back(i_dis_n);	
	desc.push_back(i_gat_n);

	ConfigurableCreator discreator(&sig_exc,&sig_inh,&d_netjocn,desc);

	D_Pattern pat_dummy;
	org.Convert
	(
		"jocndisinhibffd",
		tn,
		discreator,
		pat_dummy,
		&d_netjocn
	);

	// add the connections
	// first we need to add the p's and n's of the feedforward
	enum NetworkIndex {FFD, REV, DIS};

	DynamicSubNetwork ffd = org[FFD];
	DynamicSubNetwork rev = org[REV];
	DynamicSubNetwork dis = org[DIS];

	CircuitDescription desc_ffd = SimulationInfoBlock(ffd).DescriptionCircuit();
	CircuitDescription desc_rev = SimulationInfoBlock(rev).DescriptionCircuit();
	CircuitDescription desc_dis = SimulationInfoBlock(dis).DescriptionCircuit();

	cout << "onga bonga" << endl;
	RootReportHandler handlerjocndisinhibition("jocndisinhib.root");
	SimulationRunParameter 
		par_run
		(
			handlerjocndisinhibition,
			100000000,	// maximum number of iterations
			0.0,		// start
			1.0,		// end
			1e-3,       // report time
			1e-3,       // update time
			1e-4,       // network step time
			"jocndisinhib"
		);

	// First we going to hook up V2
	enum LayerID {V1, V2, V4, PIT, AIT};

	// Disinhib circuit id's:
	// +ve loop
	UInt_t id_in_edis_p = 0;
	UInt_t id_in_idis_p = 0;
	UInt_t id_in_gat_p = 0;
	// -ve loop
	UInt_t id_in_edis_n = 0;
	UInt_t id_in_idis_n = 0;
	UInt_t id_in_gat_n = 0;

	// Output populations
	// ffd
	UInt_t id_in_ffd_p_out = 0;
	UInt_t id_in_ffd_n_out = 0;
	// rev
	UInt_t id_in_rev_p_out = 0;
	UInt_t id_in_rev_n_out = 0;

	try {
	id_in_ffd_p_out = desc_ffd.IndexInCircuitByName("P_OUT");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "ffd - P_OUT: " << e.Description() << std::endl;
	}
	try {
	id_in_ffd_n_out = desc_ffd.IndexInCircuitByName("N_OUT");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "ffd - N_OUT: " << e.Description() << std::endl;
	}
	try {
	id_in_rev_p_out = desc_rev.IndexInCircuitByName("P_OUT");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "rev - P_OUT: " << e.Description() << std::endl;
	}
	try {
	id_in_rev_n_out = desc_rev.IndexInCircuitByName("N_OUT");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "rev - N_OUT: " << e.Description() << std::endl;
	}
	try {
	id_in_edis_p     = desc_dis.IndexInCircuitByName("e_dis_p");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "e_dis_n: " << e.Description() << std::endl;
	}
	try {
	id_in_gat_p     = desc_dis.IndexInCircuitByName("i_gat_p");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "i_gat_p: " << e.Description() << std::endl;
	}
	try {
	id_in_idis_p     = desc_dis.IndexInCircuitByName("i_dis_p");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "i_gat_p: " << e.Description() << std::endl;
	}
	try {
	id_in_edis_n     = desc_dis.IndexInCircuitByName("e_dis_n");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "e_dis_n: " << e.Description() << std::endl;
	}
	try {
	id_in_idis_n     = desc_dis.IndexInCircuitByName("i_dis_n");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "e_dis_n: " << e.Description() << std::endl;
	}
	try {
	id_in_gat_n     = desc_dis.IndexInCircuitByName("i_gat_n");
	} catch (UtilLib::GeneralException& e)
	{
		std::cout << "i_gat_n: " << e.Description() << std::endl;
	}

	const double P_to_P = 100.0;
	const double N_to_N = 100.0;
//	int layer = V2;
	DynamicSubNetwork::const_rziterator iter_rev = rev.rzbegin(V2);
	for (
			int layer = V2;
			layer != AIT;
			++layer
	)
	{

		for
		(
			DynamicSubNetwork::const_iterator iter_ffd = ffd.begin(layer), iter_dis = dis.begin(layer);
			iter_ffd != ffd.end(layer);
			++iter_ffd, ++iter_dis, ++iter_rev
		)
		{
			// Link up +ve populations
			// ffd P_OUT -> gating node
			d_netjocn.MakeFirstInputOfSecond
			(
				static_cast<NodeId>((*iter_ffd)[id_in_ffd_p_out]._id_value),
				static_cast<NodeId>((*iter_dis)[id_in_gat_p]._id_value),
				P_to_P
			);
			// ffd P_OUT -> gated node
			d_netjocn.MakeFirstInputOfSecond
			(
				static_cast<NodeId>((*iter_ffd)[id_in_ffd_p_out]._id_value),
				static_cast<NodeId>((*iter_dis)[id_in_edis_p]._id_value),
				P_to_P
			);
			// rev P_OUT -> gate inhibitory
			d_netjocn.MakeFirstInputOfSecond
			(
				static_cast<NodeId>((*iter_rev)[id_in_rev_p_out]._id_value),
				static_cast<NodeId>((*iter_dis)[id_in_idis_p]._id_value),
				P_to_P
			);
			// Link up -ve populations
			// ffd P_OUT -> gating node
			d_netjocn.MakeFirstInputOfSecond
			(
				static_cast<NodeId>((*iter_ffd)[id_in_ffd_n_out]._id_value),
				static_cast<NodeId>((*iter_dis)[id_in_gat_n]._id_value),
				N_to_N
			);
			// ffd P_OUT -> gated node
			d_netjocn.MakeFirstInputOfSecond
			(
				static_cast<NodeId>((*iter_ffd)[id_in_ffd_n_out]._id_value),
				static_cast<NodeId>((*iter_dis)[id_in_edis_n]._id_value),
				N_to_N
			);
			// rev P_OUT -> gate inhibitory
			d_netjocn.MakeFirstInputOfSecond
			(
				static_cast<NodeId>((*iter_rev)[id_in_rev_n_out]._id_value),
				static_cast<NodeId>((*iter_dis)[id_in_idis_n]._id_value),
				N_to_N
			);
		}
	}
	
	org.Configure(par_run);
	org.Evolve();

	return true;
}
