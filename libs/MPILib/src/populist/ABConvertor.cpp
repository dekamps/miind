#include <MPILib/include/populist/ABConvertor.hpp>
#include <MPILib/include/populist/AbstractCirculantSolver.hpp>
#include <MPILib/include/populist/PopulistSpecificParameter.hpp>

namespace MPILib {
namespace populist {

ABConvertor::ABConvertor( VALUE_REF_INIT
SpecialBins&, PopulationParameter& par_pop,
		PopulistSpecificParameter& par_specific, Potential& delta_v,
		Number& n_current_bins) :
		VALUE_MEMBER_INIT
		_p_specific(&par_specific), _p_pop(&par_pop), _p_n_bins(
				&n_current_bins), _p_delta_v(&delta_v) {
}

const PopulistSpecificParameter&
ABConvertor::PopSpecific() const {
	return *_p_specific;
}

const OneDMInputSetParameter&
ABConvertor::InputSet() const {
	return _param_input;
}

void ABConvertor::SortConnectionvector(const std::vector<Rate>& nodeVector,
		const std::vector<OrnsteinUhlenbeckConnection>& weightVector,
		const std::vector<NodeType>& typeVector) {
	_param_input._par_input = _scalar_product.Evaluate(nodeVector, weightVector,
			_p_pop->_tau);
	_param_input._par_input._q = _param_onedm._par_adapt._q;
}

void ABConvertor::AdaptParameters(

) {

	RecalculateSolverParameters();
}

void ABConvertor::RecalculateSolverParameters() {
	_param_input._n_current_bins = *_p_n_bins;
	_param_input._n_max_bins = _p_specific->MaxNumGridPoints();

	// current expansion factor is current number of bins
	// divided by number of initial bins

	double f = static_cast<double>(_param_input._n_current_bins)
			/ static_cast<double>(_p_specific->NrGridInitial());
	_param_input._q_expanded = f * _param_input._par_input._q;
	_param_input._t_since_rebinning = _p_pop->_tau * log(f);
	_param_input._g_max = _p_pop->_theta;
	_param_input._tau = _p_pop->_tau;
}

} /* namespace populist */
} /* namespace MPILib */
