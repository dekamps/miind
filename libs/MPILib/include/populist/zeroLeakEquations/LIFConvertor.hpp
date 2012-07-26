// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#ifndef MPILIB_POPULIST_ZEROLEAKEQUATIONS_LIFCONVERTOR_HPP_
#define MPILIB_POPULIST_ZEROLEAKEQUATIONS_LIFCONVERTOR_HPP_

#include <vector>

#include <MPILib/include/algorithm/AlgorithmInterface.hpp>

#include <MPILib/include/populist/parameters/InputParameterSet.hpp>
#include <MPILib/include/populist/parameters/OrnsteinUhlenbeckParameter.hpp>
#include <MPILib/include/populist/OrnsteinUhlenbeckConnection.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/populist/zeroLeakEquations/MuSigma.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

namespace MPILib {
namespace populist {
//Forward declarations
namespace parameters {
class PopulistSpecificParameter;
}
namespace zeroLeakEquations{
struct SpecialBins;

/**
 * @brief Converts external input from other populations in parameters that are suitable for
 * AbstractCirculantSolver and AbstractNonCirculantSolver objects.
 *
 * PopulationAlgorithm receives the contribution from its input populations as a list of pointer weight pairs.
 * Via the pointers, the input nodes can be queried for their current activation, via the weights their
 * contribution be, well, weighted. The list also contains hints as to whether for some inputs they should
 * be interpreted as Gaussian white noise, in which case they can be combined into a single input contribution in
 * a way explaned below. Internally, it maintains an instance of InputSetParameter, which contains the parameters
 * required for AbstractCirculantSolver and AbstractNonCirculantSolver to operate on.
 *
 */
class LIFConvertor {
public:

	/**
	 * Constructor registers where the output results must be written,
	 * which can happen at construction of PopulationGridController
	 */
	LIFConvertor( VALUE_REF_INIT
	SpecialBins& bins, parameters::PopulationParameter& par_pop,
			parameters::PopulistSpecificParameter& par_spec, Potential& delta_v,
			Number& n_bins

			) :
			VALUE_MEMBER_INIT
			_p_bins(&bins), _p_par_pop(&par_pop), _p_par_spec(&par_spec), _p_delta_v(
					&delta_v), _p_n_bins(&n_bins) {
	}

	/**
	 * A signaller for when the PopulationGridController starts to configure
	 * @param @todo describe this parameter
	 */
	void Configure(std::valarray<Potential>&){};

	/**
	 * This function collects the external input, and lays it out internally for use in AdaptParameters.
	 * This function must be called when input charateristics change, for example at the start of an Evolve.
	 * It used to be part of AdaptParameters, but has been moved out of there to help account for synchronous
	 * updating in networks. MPINetwork will now first go through a loop calling nodes to update their
	 * input, and then a second loop to evolve their states.
	 * @param nodeVector The vector which stores the Rates of the precursor nodes
	 * @param weightVector The vector which stores the Weights of the precursor nodes
	 * @param typeVector The vector which stores the NodeTypes of the precursor nodes
	 */
	void SortConnectionvector(const std::vector<Rate>& nodeVector,
			const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
			const std::vector<NodeType>& typeVector);

	/** This function interprets the  input contribution from other populations and converts them
	 * into input parameters for the circulant and non circulant solvers. It must be run every
	 * time _delta_v changes.
	 */
	void AdaptParameters();

	/** This function assumes that the current input is known and stored in the input parameter set,
	 * but that the parameters for the circulant and non-circulant solver is wrong. One case where
	 * this happens is after rebinning: the input firing rates and efficacies have changed, but the
	 *  solver parameters are invalid. This function recalculates them.
	 */
	void RecalculateSolverParameters();

	/**
	 * Getter for the InputParameterSet
	 * @return A reference to the input parameter Set
	 */
	parameters::InputParameterSet& getSolverParameter() {
		return _input_set;
	}
	/**
	 * Const Getter for the InputParameterSet
	 * @return A const reference to the input parameter Set
	 */
	const parameters::InputParameterSet& getSolverParameter() const {
		return _input_set;
	}

	/**
	 * Const Getter for the PopulationParameter
	 * @return A const reference to the PopulationParameter
	 */
	const parameters::PopulationParameter& getParPop() const {
		return *_p_par_pop;
	}

	/**
	 * Getter for the reversal Bins
	 * @return A reference to the reversal bins
	 */
	const Index& getIndexReversalBin() const;
	/**
	 * Getter for the current reset Bins
	 * @return A reference to the current reset Bins
	 */
	const Index& getIndexCurrentResetBin() const;
	/**
	 * @todo describe this function
	 * @return
	 */
	void UpdateRestInputParameters();

#ifdef _INVESTIGATE_ALGORITHM
	vector<ReportValue>& Values() {return _vec_value;}
#endif
private:
	/**
	 * @todo describe this function
	 * @return
	 */
	bool IsSingleDiffusionProcess(Potential h) const;
	/**
	 * @todo describe this function
	 * @return
	 */
	void SetDiffusionParameters(const MuSigma&);

	VALUE_MEMBER_REF
	/**
	 * @todo describe the members
	 */
	SpecialBins* _p_bins = nullptr;
	parameters::InputParameterSet _input_set;
	parameters::PopulationParameter* _p_par_pop = nullptr;
	parameters::PopulistSpecificParameter* _p_par_spec = nullptr;
	Potential* _p_delta_v = nullptr;
	Number* _p_n_bins = nullptr;
	Index* _p_index_reversal_bin = nullptr;
	bool _b_toggle_sort = false;
	bool _b_toggle_diffusion = false;

	std::vector<Rate> _vec_burst;
	std::vector<Rate> _vec_diffusion;
};
} /* namespace zeroLeakEquations */
} /* namespace populist */
} /* namespace MPILib */

#endif //include guard MPILIB_POPULIST_ZEROLEAKEQUATIONS_LIFCONVERTOR_HPP_
