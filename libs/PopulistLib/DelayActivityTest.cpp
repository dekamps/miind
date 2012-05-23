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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "DelayActivityTest.h"
#include "../NumtoolsLib/NumtoolsLib.h"
#include "../SparseImplementationLib/SparseImplementationLib.h"
#include "../DynamicLib/DynamicLib.h"
#include "CreateTwoPopulationNetworkCode.h"
#include "DelayActivityTest.h"
#include "OrnsteinUhlenbeckAlgorithm.h"
#include "OrnsteinUhlenbeckConnection.h"
#include "ResponseParameterBrunel.h"
#include "TestDefinitions.h"
#include <iostream>

using namespace std;
using namespace PopulistLib;
using namespace DynamicLib;
using namespace NumtoolsLib;
using namespace SparseImplementationLib;

DelayActivityTest::DelayActivityTest(boost::shared_ptr<ostream> p):
LogStream(p)
{
}

bool DelayActivityTest::Execute()
{
	if (! ResponseFunctionTest() )
		return false;
	Record("ResponseFunctionTest succeeded");

	if (! InnerProductTest() )
		return false;
	Record("InnerProductTest succeeded");

	if (! OrnsteinUhlenbeckTest() )
		return false;
	Record("OrnsteinUhlenbeckTest succeeded");

	if ( ! TwoPopulationTest() )
		return false;
	Record("TwoPopulationTest succeeded");

	if ( ! DisinhibitionDelayTest() )
		return false;
	Record("Disinhibitiondelaytest succeeded");

	return true;
}

bool DelayActivityTest::ResponseFunctionTest()
{
	// These are points on a curve in figure 3
	// of Brunel, N. (2000), Network: Comput. Neural Syst. 11 (2000) 261-280.

	double epsilon = 1e-5;

	ResponseParameterBrunel parameter = RESPONSE_CURVE_PARAMETER;

	for (Index index_point = 0; index_point < NUMBER_RESPONSE_CURVE_POINTS; index_point++)
	{
		parameter.mu = MU[index_point];     // mV
		if (
			! IsApproximatelyEqualTo
			(
				ResponseFunction(parameter),
				RESPONSE_NON_REFRACTIVE[index_point],
				epsilon
			) 
		)
			return false;
	}

	return true;
}

bool DelayActivityTest::InnerProductTest()
{

	OU_DynamicNode node1;
	OU_DynamicNode node2;
	OU_DynamicNode node3;

	node1.SetValue(1.0);
	node2.SetValue(1.0);
	node3.SetValue(1.0);

	typedef pair<AbstractSparseNode<double,OrnsteinUhlenbeckConnection>*,OrnsteinUhlenbeckConnection> connection;
	typedef OU_DynamicNode::predecessor_iterator predecessor_iterator;

	vector<connection> vector_of_connections;

	OrnsteinUhlenbeckConnection connection_1(1, 1);
	OrnsteinUhlenbeckConnection connection_2(1, 2);
	OrnsteinUhlenbeckConnection connection_3(1, 3);

	vector_of_connections.push_back(connection(&node1,connection_1));
	vector_of_connections.push_back(connection(&node2,connection_2));
	vector_of_connections.push_back(connection(&node3,connection_3));

	// Need a concrete Algorithm to calculate the inner product

	OrnsteinUhlenbeckParameter parameter;
	OrnsteinUhlenbeckAlgorithm algorithm(parameter);

	// Cumbersome notation ncessary, because this Algorithm isn't part of a node

	double f_inner_product = 
		algorithm.InnerProduct
		(
			predecessor_iterator( &(*vector_of_connections.begin()) ),
			predecessor_iterator( &(*vector_of_connections.begin()) + vector_of_connections.size() )
		);

	if (
		! NumtoolsLib::IsApproximatelyEqualTo
		(
			f_inner_product,
			6.0,
			1e-10
		) 
	)
		return false;

	double f_squared_inner_product = 
		algorithm.InnerSquaredProduct
		(
			predecessor_iterator( &(*vector_of_connections.begin()) ),
			predecessor_iterator( &(*vector_of_connections.begin()) + vector_of_connections.size() )
		);
	if (
		! NumtoolsLib::IsApproximatelyEqualTo
		(
			f_squared_inner_product,
			14.0,
			1e-10
		)
	)
		return false;

	return true;
}

bool DelayActivityTest::OrnsteinUhlenbeckTest()
{
	OU_Network network;

	// Reconstruct the first point of the response curve

	OrnsteinUhlenbeckParameter parameter;

	parameter._tau            = RESPONSE_CURVE_PARAMETER.tau;
	parameter._tau_refractive = RESPONSE_CURVE_PARAMETER.tau_refractive;
	parameter._V_reset        = RESPONSE_CURVE_PARAMETER.V_reset;
	parameter._theta          = RESPONSE_CURVE_PARAMETER.theta;
	parameter._V_reversal     = 0;

	OrnsteinUhlenbeckAlgorithm algorithm(parameter);

	// Let's create RateAlgorithms for the first mu point.
	Rate rate_excitatory = test_rate_e(MU[0]);
	OU_RateAlgorithm ExcitatoryInput(rate_excitatory);

	Rate rate_inhibitory = test_rate_i(MU[0]);
	OU_RateAlgorithm InhibitoryInput(rate_inhibitory);

	// Add them into the network 
	NodeId id_excitatory  = network.AddNode(ExcitatoryInput,EXCITATORY);
	NodeId id_inhibitory  = network.AddNode(InhibitoryInput,INHIBITORY);
	NodeId id_node        = network.AddNode(algorithm,EXCITATORY);

	// Set the weights appropriately
	bool b_weight = true;
	OrnsteinUhlenbeckConnection 
		EE
		(
			N_EXC, 
			J_EE
		);

	OrnsteinUhlenbeckConnection 
		EI
		(
			N_INH, 
			-J_EI
		);

	b_weight &= 
		network.MakeFirstInputOfSecond
		(
			id_excitatory,
			id_node, 
			EE
		);

	b_weight &= 
		network.MakeFirstInputOfSecond
		(
			id_inhibitory,
			id_node, 
			EI
		);

	if (! b_weight )
		return false;

	SimulationRunParameter run_parameter = TEST_ORNSTEIN_RUN_PARAMETER;
	
	if ( network.ConfigureSimulation(run_parameter) )
		return network.Evolve();
	else
		return false;


	return true;
}


bool DelayActivityTest::TwoPopulationTest()
{

	NodeId id_cortical_background;
	NodeId id_excitatory_main;
	NodeId id_inhibitory_main;
	NodeId id_rate;
	OU_Network network = 
		CreateTwoPopulationNetwork<OrnsteinUhlenbeckAlgorithm, DynamicNetworkImplementation<OU_Connection> >
		(
			&id_cortical_background,
			&id_excitatory_main,
			&id_inhibitory_main,
			&id_rate,
			TWOPOPULATION_NETWORK_EXCITATORY_PARAMETER,
			TWOPOPULATION_NETWORK_INHIBITORY_PARAMETER
		);

	RootReportHandler 
		handler
		(
			STRING_TEST_DIR + STR_TWOPOPULATION_DELAY, // root file name
			false,       // don't display canvas
			true         // store in root file
		);

	handler.AddNodeToCanvas(id_excitatory_main);
	handler.AddNodeToCanvas(id_inhibitory_main);

	const SimulationRunParameter 
		run_disinhibition_parameter
		(
			handler,
			NUMBER_INTEGRATION_MAXIMUM,
			TWOPOPULATION_TIME_BEGIN,
			TWOPOPULATION_TIME_END,
			TWOPOPULATION_TIME_REPORT,
			TWOPOPULATION_TIME_UPDATE,
			TWOPOPULATION_TIME_NETWORK,
			"test/twopopulationtest.log"
		);

	bool b_configure = network.ConfigureSimulation
			(
				run_disinhibition_parameter
			);

	if (! b_configure )
		return false;


	bool b_evolve = network.Evolve();

	return b_evolve;
	return true;
}

bool DelayActivityTest::DisinhibitionDelayTest()
{
	try {

		OU_Network network;

		// Create the necessary algorithms

		RateAlgorithm<OrnsteinUhlenbeckConnection> alg_cortical_bg(DELAY_RATE_CORTICAL_BG);

		OrnsteinUhlenbeckAlgorithm 
			alg_excitatory
			(
				DELAY_EXCITATORY_PARAMETER
			);

		OrnsteinUhlenbeckAlgorithm 
			alg_inhibitory
			(
				DELAY_INHIBITORY_PARAMETER
			);
						
		RateFunctor<OrnsteinUhlenbeckConnection> pulse(Pulse);

		// Add all Nodes
		NodeId id_cortical_background = 
			network.AddNode
			(
				alg_cortical_bg, 
				EXCITATORY
			);


		NodeId id_excitatory_main =
			network.AddNode
			(
				alg_excitatory,
				EXCITATORY
			);
				

		NodeId id_inhibitory_main =
			network.AddNode
			(
				alg_inhibitory,
				INHIBITORY
			);

		NodeId id_suppressor =
			network.AddNode
			(
				alg_inhibitory,
				INHIBITORY
			);

		NodeId id_control =
			network.AddNode
			(
				pulse,
				EXCITATORY
			);
                

		NodeId id_delay =
			network.AddNode
			(
				alg_excitatory,
				EXCITATORY
			);

		NodeId id_dis =
			network.AddNode
			(
				alg_inhibitory,
				INHIBITORY
			);
		// Adding connections:

		// Adding connections to excitatory main:
		OrnsteinUhlenbeckConnection 
			bg_to_E
			(
				DELAY_X*DELAY_C_E,
				DELAY_J_EE
			);

		bool b_connection = 
			network.MakeFirstInputOfSecond
			(
				id_cortical_background,
				id_excitatory_main,
				bg_to_E
			);

		OrnsteinUhlenbeckConnection 
			I_to_E
			(
				(1 - DELAY_X_SUP)*DELAY_C_I,
				-DELAY_J_EI
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_inhibitory_main,
				id_excitatory_main,
				I_to_E
			);

		OrnsteinUhlenbeckConnection 
			E_to_E
			(
				DELAY_X*(1 - DELAY_X_DA)*DELAY_C_E,
				DELAY_J_EE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_excitatory_main,
				id_excitatory_main,
				E_to_E
			);

		OrnsteinUhlenbeckConnection 
			DA_to_E
			(
				DELAY_X*DELAY_X_DA*DELAY_C_E,
				DELAY_J_EE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_delay, 
				id_excitatory_main,
				DA_to_E
			);

		OrnsteinUhlenbeckConnection 
			SUP_to_E
			(
				DELAY_X_SUP*DELAY_C_I,
				-DELAY_J_EI
			);
		
		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_suppressor,
				id_excitatory_main,
				SUP_to_E
			);

		// Adding connections to inhibitory main:

		OrnsteinUhlenbeckConnection 
			bg_to_I
			(
				DELAY_X*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_cortical_background,
				id_inhibitory_main,
				bg_to_I
			);
										
		OrnsteinUhlenbeckConnection 
			E_to_I
			(
				DELAY_X*(1 - DELAY_X_DA)*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_excitatory_main,
				id_inhibitory_main,
				E_to_I
			);

		OrnsteinUhlenbeckConnection 
			I_to_I
			(
				(1 - DELAY_X_SUP)*DELAY_C_I,
				-DELAY_J_II
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_inhibitory_main,
				id_inhibitory_main,
				I_to_I
			);

		OrnsteinUhlenbeckConnection 
			DA_to_I
			(
				DELAY_X*DELAY_X_DA*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_delay,
				id_inhibitory_main,
				DA_to_I
			);

		OrnsteinUhlenbeckConnection 
			SUP_to_I
			(
				DELAY_X_SUP*DELAY_C_I,
				-DELAY_J_II
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_suppressor,
				id_inhibitory_main,
				SUP_to_I
			);

		// Adding connections to SUP

		OrnsteinUhlenbeckConnection 
			bg_to_sup
			(
				DELAY_X*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_cortical_background,
				id_suppressor,
				bg_to_sup
			);

		OrnsteinUhlenbeckConnection 
			E_to_SUP
			(
				DELAY_X*(1 - DELAY_X_DA)*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_excitatory_main,
				id_suppressor,
				E_to_SUP
			);

		OrnsteinUhlenbeckConnection 
			I_to_SUP
			(
				DELAY_C_I, // Take DIS =0 ,not 5 Hz
				-DELAY_J_II
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_inhibitory_main,
				id_suppressor,
				I_to_SUP
			);

		OrnsteinUhlenbeckConnection 
			DIS_to_SUP
			(
				DELAY_X_DIS*DELAY_C_I,
				-DELAY_GAMMA_DIS*DELAY_J_II
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_dis,
				id_suppressor,
				DIS_to_SUP
			);

		OrnsteinUhlenbeckConnection 
			DA_to_SUP
			(
				DELAY_X*DELAY_X_DA*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_delay,
				id_suppressor,
				DA_to_SUP
			);
									
		// Adding Nodes to DA
		OrnsteinUhlenbeckConnection 
			bg_to_DA
			(
				DELAY_X*DELAY_C_E,
				DELAY_J_EE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_cortical_background,
				id_delay,
				bg_to_DA
			);

		OrnsteinUhlenbeckConnection 
			E_to_DA
			(
				DELAY_X*DELAY_C_E,
				DELAY_GAMMA_DA*DELAY_J_EE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_excitatory_main,
				id_delay,
				E_to_DA
			);

		OrnsteinUhlenbeckConnection 
			I_to_DA
			(
				(1 - DELAY_X_SUP)*DELAY_C_I,
				-DELAY_J_EI
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_inhibitory_main,				
				id_delay,
				I_to_DA
			);

		OrnsteinUhlenbeckConnection 
			SUP_to_DA
			(
				DELAY_X_SUP*DELAY_C_I,
				-DELAY_GAMMA_SUP*DELAY_J_EI
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_suppressor,
				id_delay,
				SUP_to_DA
			);

		// Adding Nodes To DIS

		OrnsteinUhlenbeckConnection 
			bg_to_DIS
			(
				DELAY_X*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_cortical_background,
				id_dis,
				bg_to_DIS
			);
	

		OrnsteinUhlenbeckConnection 
			E_to_DIS
			(
				DELAY_X*( 1 - DELAY_X_C - DELAY_X_DAC)*DELAY_C_E,
				DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_excitatory_main,
				id_dis,
				E_to_DIS
			);

		OrnsteinUhlenbeckConnection 
			C_to_DIS
			(
				DELAY_C_E*DELAY_X_C,
				DELAY_GAMMA_C*DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_control,
				id_dis,
				C_to_DIS
			);
										
		OrnsteinUhlenbeckConnection 
			I_to_DIS
			(
				DELAY_C_I,
				-DELAY_GAMMA_I_DIS*DELAY_J_II
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
				id_inhibitory_main,
				id_dis,
				I_to_DIS
			);

		OrnsteinUhlenbeckConnection 
			DA_to_DIS          
			(
				DELAY_X*DELAY_X_DAC*DELAY_C_E,
                                DELAY_GAMMA_DA_DIS*DELAY_J_IE
			);

		b_connection &= 
			network.MakeFirstInputOfSecond
			(
			        id_delay,
				id_dis,
				DA_to_DIS
			);

		if (b_connection)
		{
			string file_path = STRING_TEST_DIR + STR_DELAY_DISINHIBITION_NAME;
			RootReportHandler 
				handler
				(
					file_path.c_str(), 
					false, 
					true
				);

			handler.AddNodeToCanvas
			(
				id_delay
			);

			SimulationRunParameter 
				parameter
				(
					handler,
					100000,
					0,
					1,
					5e-3,
					5e-2,
					1e-4,
					"test/disinhibitiondelaytest.log"
				);

			if ( network.ConfigureSimulation(parameter) )
				network.Evolve();
			else
				return false;

		}
			else
				cout << "Wrong connection sign" << endl;
	}

	catch (NetLib::NetLibException& exception)
	{
		cout << exception.Description() << endl;
	}
	catch (UtilLib::GeneralException& exception)
	{
		cout << exception.Description() << endl;
	}
	catch (...)
	{
		cout << "Unknown error occured" << endl;
	}
	return true;
}

