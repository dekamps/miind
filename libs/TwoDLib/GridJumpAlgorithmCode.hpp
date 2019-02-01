#ifndef _CODE_LIBS_TWODLIBLIB_GRIDJUMPALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIBLIB_GRIDJUMPALGORITHMCODE_INCLUDE_GUARD

#include "GridJumpAlgorithm.hpp"

namespace TwoDLib {

  GridJumpAlgorithm::GridJumpAlgorithm
  (
    const std::string& model_name,
		const std::string& transform_matrix,
		MPILib::Time h,
		double start_v,
		double start_w,
		MPILib::Time tau_refractive,
		const std::string&  rate_method
  ) : GridAlgorithm(model_name,transform_matrix,h,start_v,start_w,tau_refractive,rate_method)
  {}

  GridJumpAlgorithm::GridJumpAlgorithm(const GridJumpAlgorithm& rhs)
  : GridAlgorithm(rhs)
  {}

  GridJumpAlgorithm* GridJumpAlgorithm::clone() const
	{
	  return new GridJumpAlgorithm(*this);
	}

  void GridJumpAlgorithm::setupMasterSolver(double cell_width){
		try {
			std::unique_ptr<MasterGridJump> p_master(new MasterGridJump(_sys,cell_width));
			_p_master_jump = std::move(p_master);
		}
		// TODO: investigate the following
		// for some reason, the exception is usually not caught by the main program, which is why we write its message to cerr here.
		catch(TwoDLibException& e){
			std::cerr << e.what() << std::endl;
			throw e;
		}

	}

  void GridJumpAlgorithm::applyMasterSolver(std::vector<MPILib::Rate> rates) {
			_p_master_jump->Apply(_n_steps*_dt,rates);
	}

  void GridJumpAlgorithm::prepareEvolve
	(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<CustomConnectionParameters>& weightVector,
		const std::vector<MPILib::NodeType>& typeVector
	)
	{
		if (_efficacy_map.size() == 0){
			FillMap(weightVector);
		}

		// take into account the number of connections

		assert(nodeVector.size() == weightVector.size());
		for (MPILib::Index i = 0; i < nodeVector.size(); i++){
			_vec_vec_delay_queues[0][i].updateQueue(nodeVector[i]*weightVector[i]._params.at("num_connections"));
		}

	}

  void GridJumpAlgorithm::FillMap(const std::vector<CustomConnectionParameters>& vec_weights)
	{
		// this function will only be called once;
		_efficacy_map = std::vector<double>(vec_weights.size());
    _connection_stat_v = std::vector<double>(vec_weights.size());

 		for(MPILib::Index i_weight = 0; i_weight < _efficacy_map.size(); i_weight++){
			_efficacy_map[i_weight] = vec_weights[i_weight]._params.at("efficacy");
      _connection_stat_v[i_weight] = vec_weights[i_weight]._params.at("stationary");
		}

    _p_master_jump->CalculateStaticEfficiaciesForConductance(_efficacy_map, _connection_stat_v);

		_vec_vec_delay_queues = std::vector< std::vector<MPILib::DelayedConnectionQueue> >(0); // MeshAlgorithm really only uses the first array, i.e. the rates it receives in prepareEvole
 		_vec_vec_delay_queues.push_back( std::vector<MPILib::DelayedConnectionQueue>(vec_weights.size()));
		for(unsigned int q = 0; q < vec_weights.size(); q++){
			_vec_vec_delay_queues[0][q] = MPILib::DelayedConnectionQueue(_dt, vec_weights[q]._params.at("delay"));
		}
	}

}

#endif
