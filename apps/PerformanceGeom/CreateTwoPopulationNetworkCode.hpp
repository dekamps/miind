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
#ifndef _CODE_LIBS_POPULISTLIB_CREATETWOPOPULATIONNETWORKCODE_INCLUDE_GUARD
#define _CODE_LIBS_POPULISTLIB_CREATETWOPOPULATIONNETWORKCODE_INCLUDE_GUARD

#include "CreateTwoPopulationNetwork.hpp"
#include <MPILib/include/populist/RateFunctorCode.hpp>
namespace PerformanceGeom {

	inline Rate CorticalBackground(Time t)
	{
		return RATE_TWOPOPULATION_EXCITATORY_BACKGROUND;
	}

  template <class Algorithm>
  Network CreateTwoPopulationNetwork
  (	
   MPILib::NodeId* p_id_cortical_background,
   MPILib::NodeId* p_id_excitatory_main,
   MPILib::NodeId* p_id_inhibitory_main,
   MPILib::NodeId* p_id_rate,
   const typename Algorithm::Parameter& par_exc,
   const typename Algorithm::Parameter& par_inh,
   MPILib::Time    t_delay_e,
   MPILib::Time    t_delay_i
  ) 
  {

    typename Algorithm::WeightType 
      	connection_unit
      	(
	 1.0, 
	 1.0
      	);

    typename Algorithm::WeightType 
       	connection_min_unit
      	(
	 1.0, 
	 -1.0
      	);

      	Network network;

       	// Create cortical background, and add to network
	MPILib::populist::RateFunctor<typename Algorithm::WeightType> cortical_background(CorticalBackground);
       	*p_id_cortical_background = network.addNode(cortical_background, MPILib::EXCITATORY_GAUSSIAN);

       	// Create excitatory main population
       	Algorithm algorithm_excitatory(par_exc);
       	*p_id_excitatory_main = network.addNode(algorithm_excitatory, MPILib::EXCITATORY_GAUSSIAN);

	
	// Create inhibitory main population
       	Algorithm algorithm_inhibitory(par_inh);
       	*p_id_inhibitory_main = network.addNode(algorithm_inhibitory, MPILib::INHIBITORY_GAUSSIAN);

       	// Background and excitatory connection only differ in x, 1 - x
       	typename Algorithm::WeightType 
	  connection_J_EE_BG
	  (
	   TWOPOPULATION_C_E*(1-TWOPOPULATION_FRACTION), 
	   TWOPOPULATION_J_EE
	  );

       	network.makeFirstInputOfSecond
   	(
	 *p_id_cortical_background,
	 *p_id_excitatory_main,
	 connection_J_EE_BG
       	);

       	// Excitatory connection to itself

       	typename Algorithm::WeightType 
       	connection_J_EE
      	(
	 TWOPOPULATION_C_E*TWOPOPULATION_FRACTION,
   	 TWOPOPULATION_J_EE
      	);

     	network.makeFirstInputOfSecond
       	(
	 *p_id_excitatory_main,
	 *p_id_excitatory_main,
	 connection_J_EE
	 );

       	// Background connection to I

       	typename Algorithm::WeightType 
      	connection_J_IE_BG
      	(
	 static_cast<Number>(TWOPOPULATION_C_E*(1 - TWOPOPULATION_FRACTION)),
	 TWOPOPULATION_J_IE
      	);

       	network.makeFirstInputOfSecond
       	(
	 *p_id_cortical_background,
	 *p_id_inhibitory_main,
	 connection_J_IE_BG
       	);

       	// E to I
       	typename Algorithm::WeightType 
      	connection_J_IE
       	(
	 static_cast<Number>(TWOPOPULATION_C_E*TWOPOPULATION_FRACTION),
	 TWOPOPULATION_J_IE
      	);

	network.makeFirstInputOfSecond
	(
	 *p_id_excitatory_main,
	 *p_id_inhibitory_main,
	 connection_J_IE
	 );

       	// I to E
       	typename Algorithm::WeightType 
      	connection_J_EI
       	(
	 TWOPOPULATION_C_I,
	 -TWOPOPULATION_J_EI
       	);
       	network.makeFirstInputOfSecond
       	(
	 *p_id_inhibitory_main,
	 *p_id_excitatory_main,
	 connection_J_EI
       	);

       	// I to I
       	typename Algorithm::WeightType 
       	connection_J_II
       	(
	 TWOPOPULATION_C_I,
	 -TWOPOPULATION_J_II
       	);

       	network.makeFirstInputOfSecond
       	(			
	 *p_id_inhibitory_main,
	 *p_id_inhibitory_main,
	 connection_J_II
       	);

	return network;
  }
}

#endif // include guard
