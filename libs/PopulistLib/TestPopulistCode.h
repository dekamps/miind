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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_POPULISTLIB_TESTPOPULISTCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_TESTPOPULISTCODE_INCLUDE_GUARD

#include "TestPopulist.h"
#include "TestDefinitions.h"
#include "TestResponseCurveDefinitions.h"
#include "PopulationAlgorithmCode.h"
#include "Response.h"

namespace PopulistLib { 


	template <class ZeroLeakEquations>
	bool TestPopulist::ResponseCurveDouble
	(
		bool						b_refractive,
		bool						b_fit, 
		bool						b_polynomial,
		Index						i,
		PopulistLib::Potential		sigma,
		Number						n_dif
	) const
	{
		Pop_Network					network;
		PopulistSpecificParameter	par_spec;
		Rate						input_rate_exc;
		Rate						input_rate_inh;
		NodeId						id;

		ResponseCurveDoubleNetwork<ZeroLeakEquations>
		(
			&network,
			&par_spec,
			b_fit,
			b_polynomial,
			&input_rate_exc,
			&input_rate_inh,
			&id,
			n_dif
		);
	
	
		string handlername;
		string logname;				

		ResponseSingleRunParameter
		(
			b_refractive,
			b_fit,
			b_polynomial,
			true,
			i,
			handlername,
			logname,
			&par_spec,
			RESPONSE_CURVE_SINGLE_NBINS
		); 
	

		RootReportHandler 
			handler
			(
				handlername.c_str(),
				ONSCREEN,
				INFILE,
				RESPONSE_CURVE_T_END
			);
	
		SimulationRunParameter
		par_run
		(
			handler,
			RESPONSE_CURVE_MAX_ITER,
			RESPONSE_CURVE_T_BEGIN,
			RESPONSE_CURVE_T_END,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_NETWORK,
			logname
		);

		par_run.Handler().AddNodeToCanvas(id);

		bool b_configure =
			network.ConfigureSimulation(par_run);

		if (! b_configure)
			return false;

		if ( network.begin(id) != network.end(id) ){
			Pop_Network::predecessor_iterator iter = network.begin(id);

			// precise values don't matter, as long as they reproduce correct mu and sigma
			double h_e = 1;
			double N_e = (MU[i] + SIGMA*SIGMA)/(2*PARAMETER_NEURON._tau);
			PopulationConnection con_e(N_e,h_e);
			iter.SetWeight(con_e);
			input_rate_exc = 1;

			double h_i = -1;
			double N_i = (SIGMA*SIGMA -MU[i])/(2*PARAMETER_NEURON._tau);
			PopulationConnection  con_i(N_i,h_i);
			iter++;
			iter.SetWeight(con_i);
			input_rate_inh = 1;
		}
		bool b_evolve = network.Evolve();
		if (! b_evolve)
			return false;

		return true;
	}

	template <class ZeroLeakEquations>
	void TestPopulist::ResponseCurveSingleNetwork
	(
		Pop_Network*				p_net,
		PopulistSpecificParameter*	p_par_spec,
		bool						b_refraction,
		bool						b_fit,
		bool						b_polynomial,
		Rate*						p_rate_exc,
		NodeId*						p_id
	) const
	{

		ResponseSpecificParameter
		(
			b_fit,
			b_polynomial,
			false,
			p_par_spec
		);
	
		PopulationParameter par_neuron = PARAMETER_NEURON;
		if (! b_refraction)
			par_neuron._tau_refractive = 0.0;

		// Define the node
		PopulationAlgorithm_<PopulationConnection> 
			the_algorithm
			(
				PopulistParameter
				(
					par_neuron, 
					*p_par_spec
				)
			);

		*p_id = 
			p_net->AddNode 
			(
				the_algorithm,
				EXCITATORY
			);

		// Define an input rate population
		RateAlgorithm <PopulationConnection> rate_input_excitatory(p_rate_exc);
	
		NodeId id_exc = 
			p_net->AddNode
			(
				rate_input_excitatory,
				EXCITATORY_BURST
			);

		PopulationConnection 
			connection
			(
				0, 
				0
			);

		p_net->MakeFirstInputOfSecond
		(
			id_exc, 
			*p_id, 
			connection
		);
	}

	template <class ZeroLeakEquations>
	void TestPopulist::ResponseCurveDoubleNetwork
	(
		Pop_Network*				p_net,
		PopulistSpecificParameter*	p_par_spec,
		bool						b_refractive,
		bool						b_fit,
		bool						b_polynomial,
		Rate*						p_rate_exc,
		Rate*						p_rate_inh,
		NodeId*						p_id
	) const
	{	
		// Define an rate population
		RateAlgorithm < PopulationConnection > 
			rate_exc(p_rate_exc);
	
		RateAlgorithm< PopulationConnection >
			rate_inh(p_rate_inh);

		NodeId id_exc = 
			p_net->AddNode
			(
				rate_exc,
				EXCITATORY
			);

		NodeId id_inh =
			p_net->AddNode
			(
				rate_inh,
				INHIBITORY
			);

		// Define the node
		ResponseSpecificParameter
		(
			b_fit,
			b_polynomial,
			true,
			p_par_spec
		);
 
		PopulationParameter par_neur = PARAMETER_NEURON;
		if (! b_refractive)
			par_neur._tau_refractive = 0.0;

		PopulationAlgorithm_<PopulationConnection> 
			the_algorithm
			(
				PopulistParameter
				(
					par_neur,
					*p_par_spec
				)
			);

		*p_id = 
			p_net->AddNode 
			(
				the_algorithm,
				EXCITATORY
			);

		PopulationConnection 
			connection_1
			(
				0, 
				0
			);

		PopulationConnection
			connection_2
			(	
				0,
				0
			);

		p_net->MakeFirstInputOfSecond
		(
			id_exc, 
			*p_id, 
			connection_1
		);

		p_net->MakeFirstInputOfSecond
		(
			id_inh,
			*p_id,
			connection_2
		);
	}

	template <class ZeroLeakEquations>
	bool TestPopulist::ResponseCurveSingle
	(
	 bool b_refractive,
		bool					b_fit,
		bool					b_polynomial,
		Index					i,
		PopulistLib::Potential	sigma
	) const
	{

		PopulistSpecificParameter par_spec;
		Pop_Network		network;
		Rate			rate_exc;
		NodeId			id;

		ResponseCurveSingleNetwork<ZeroLeakEquations>
		(
			&network,
			&par_spec,
			b_fit,
			b_polynomial,
			&rate_exc,
			&id
		);

		SinglePopulationInput inp;
		inp =  CovertMuSigma(MU[i], sigma, PARAMETER_NEURON);

		// Set the value for the RateAlgorithm
		rate_exc = inp._rate;

		string handlername;
		string logname;				

		ResponseSingleRunParameter
		(
		        b_refractive,
			b_fit,
			b_polynomial,
			false,
			i,
			handlername,
			logname,
			&par_spec,
			RESPONSE_CURVE_SINGLE_NBINS
		); 
	

		RootReportHandler 
			handler
			(
				handlername.c_str(),
				ONSCREEN,
				INFILE,
				RESPONSE_CURVE_T_END
			);

		SimulationRunParameter
		par_run
		(
			handler,
			RESPONSE_CURVE_MAX_ITER,
			RESPONSE_CURVE_T_BEGIN,
			RESPONSE_CURVE_T_END,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_REPORT,
			RESPONSE_CURVE_T_NETWORK,
			logname
		);

		bool b_configure =
			network.ConfigureSimulation(par_run);

		if (! b_configure)
			return false;

		// Seek out the connection from the rate algorithm and set the weight
		PopulationConnection con(1,inp._h);
		Pop_Network::predecessor_iterator iter = network.begin(id);
		iter.SetWeight(con);	

		par_run.Handler().AddNodeToCanvas(id);

		bool b_evolve = network.Evolve();
		if (! b_evolve)
			return false;

		return true;
	}

} // end of namespace


#endif // include guard
