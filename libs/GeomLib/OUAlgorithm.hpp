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

#ifndef MPILIB_ALGORITHMS_OUALGORITHM_HPP_
#define MPILIB_ALGORITHMS_OUALGORITHM_HPP_

#include <NumtoolsLib/NumtoolsLib.h>
#include <MPILib/algorithm/WilsonCowanParameter.hpp>
#include <MPILib/algorithm/AlgorithmInterface.hpp>
#include <MPILib/include/DelayedConnection.hpp>
#include "GeomInputConvertor.hpp"
#include "MuSigmaScalarProduct.hpp"
#include "NeuronParameter.hpp"
#include "ResponseParameter.hpp"

using MPILib::algorithm::AlgorithmGrid;
using MPILib::algorithm::AlgorithmInterface;
using MPILib::DelayedConnection;
using MPILib::Rate;
using MPILib::SimulationRunParameter;
using MPILib::Time;
using NumtoolsLib::DVIntegrator;

namespace GeomLib {

  //!< \brief Rate-based model with gain function based on a diffusion process.

 
  class OUAlgorithm : public AlgorithmInterface<MPILib::DelayedConnection> {
	public:


    /// Create an OUAlgorithm from neuronal parameters 
    OUAlgorithm
    (
     const NeuronParameter&
     );

    /// copy ctor
    OUAlgorithm
    (
     const OUAlgorithm&
     );

    /// virtual destructor
    virtual ~OUAlgorithm();

    /// configure algorithm
    virtual void configure
    (
     const SimulationRunParameter& //!< simulation run parameter
     );

    /**                                                                                                                                                                          
     * Evolve the node state                                                                                                                                                     
     * @param nodeVector Vector of the node States                                                                                                                               
     * @param weightVector Vector of the weights of the nodes                                                                                                                    
     * @param time Time point of the algorithm                                                                                                                                   
     */
    virtual void evolveNodeState
                  (
		   const std::vector<Rate>& nodeVector,
		   const std::vector<MPILib::DelayedConnection>& weightVector, 
		   Time time
		   );

	/**
	 * prepare the Evolve method
	 * @param nodeVector Vector of the node States
	 * @param weightVector Vector of the weights of the nodes
	 * @param typeVector Vector of the NodeTypes of the precursors
	 */
	virtual void prepareEvolve(const std::vector<MPILib::Rate>& nodeVector,
			const std::vector<MPILib::DelayedConnection>& weightVector,
			const std::vector<MPILib::NodeType>& typeVector);



    /// Current AlgorithmGrid
    virtual AlgorithmGrid getGrid(MPILib::NodeId) const;

    /**                                                                                                                                                                          
     * Cloning operation, to provide each DynamicNode with its own                                                                                                               
     * Algorithm instance. Clients use the naked pointer at their own risk.                                                                                                      
     */
    virtual OUAlgorithm* clone() const;

    //! Current tme of the simulation
    virtual Time getCurrentTime() const;
    
    //! Current output rate of the population
    virtual Rate getCurrentRate() const;


  private:

    AlgorithmGrid InitialGrid() const;
    vector<double> InitialState() const;

    ResponseParameter 
     InitializeParameters
    (
     const NeuronParameter& 
     ) const;

    NeuronParameter                 _parameter_neuron;
    ResponseParameter     			_parameter_response;
    DVIntegrator<ResponseParameter>	_integrator;
    MuSigmaScalarProduct<MPILib::DelayedConnection> _scalar_product;
  };

}  // GeomLib

#endif // include guard
