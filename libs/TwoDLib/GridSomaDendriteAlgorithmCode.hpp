#ifndef _CODE_LIBS_TWODLIBLIB_GRIDSOMADENDRITEALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIBLIB_GRIDSOMADENDRITEALGORITHMCODE_INCLUDE_GUARD

#include "GridSomaDendriteAlgorithm.hpp"

namespace TwoDLib {

  GridSomaDendriteAlgorithm::GridSomaDendriteAlgorithm
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

  GridSomaDendriteAlgorithm::GridSomaDendriteAlgorithm(const GridSomaDendriteAlgorithm& rhs)
  : GridAlgorithm(rhs)
  {}

  GridSomaDendriteAlgorithm* GridSomaDendriteAlgorithm::clone() const
	{
	  return new GridSomaDendriteAlgorithm(*this);
	}

  void GridSomaDendriteAlgorithm::setupMasterSolver(double cell_width){
		try {
			std::unique_ptr<MasterGridSomaDendrite> p_master(new MasterGridSomaDendrite(_sys,cell_width));
			_p_master_jump = std::move(p_master);
		}
		// TODO: investigate the following
		// for some reason, the exception is usually not caught by the main program, which is why we write its message to cerr here.
		catch(TwoDLibException& e){
			std::cerr << e.what() << std::endl;
			throw e;
		}

	}

  void GridSomaDendriteAlgorithm::applyMasterSolver(std::vector<MPILib::Rate> rates) {
			_p_master_jump->Apply(_n_steps*_dt,rates);
	}

  void GridSomaDendriteAlgorithm::prepareEvolve
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
      if(_connection_types[i] == "SomaDendrite"){
        _connection_stat_v[i] = nodeVector[i];
        _vec_vec_delay_queues[0][i].updateQueue(10000.0*std::stod(weightVector[i]._params.at("num_connections")));
      } else {
        double offset = 0.0;
        if (weightVector[i]._params.find("avgv_offset") != weightVector[i]._params.end())
          offset = std::stod(weightVector[i]._params.at("avgv_offset"));

        _vec_vec_delay_queues[0][i].updateQueue((offset+nodeVector[i])*std::stod(weightVector[i]._params.at("num_connections")));
      }
		}

    _p_master_jump->CalculateDynamicEfficiacies(_connection_types, _efficacy_map, _connection_stat_v, _connection_conductances);

	}

  void GridSomaDendriteAlgorithm::FillMap(const std::vector<CustomConnectionParameters>& vec_weights)
	{
		// this function will only be called once;
		_efficacy_map = std::vector<double>(vec_weights.size());
    _connection_stat_v = std::vector<double>(vec_weights.size());
    _connection_conductances = std::vector<double>(vec_weights.size());
    _connection_types = std::vector<std::string>(vec_weights.size());

 		for(MPILib::Index i_weight = 0; i_weight < _efficacy_map.size(); i_weight++){
			_efficacy_map[i_weight] = std::stod(vec_weights[i_weight]._params.at("efficacy"));
      _connection_conductances[i_weight] = std::stod(vec_weights[i_weight]._params.at("conductance"));
      _connection_types[i_weight] = vec_weights[i_weight]._params.at("type");
		}

    _p_master_jump->InitializeEfficacyVectors(_efficacy_map.size());

		_vec_vec_delay_queues = std::vector< std::vector<MPILib::DelayedConnectionQueue> >(0); // MeshAlgorithm really only uses the first array, i.e. the rates it receives in prepareEvole
 		_vec_vec_delay_queues.push_back( std::vector<MPILib::DelayedConnectionQueue>(vec_weights.size()));
		for(unsigned int q = 0; q < vec_weights.size(); q++){
			_vec_vec_delay_queues[0][q] = MPILib::DelayedConnectionQueue(_network_time_step, std::stod(vec_weights[q]._params.at("delay")));
		}
	}

}

#endif
