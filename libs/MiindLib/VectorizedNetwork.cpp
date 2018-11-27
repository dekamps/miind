#include "VectorizedNetwork.hpp"
#include <boost/timer/timer.hpp>

using namespace MiindLib;

VectorizedNetwork::VectorizedNetwork(MPILib::Time time_step):
_num_nodes(0),
_n_steps(10),
_network_time_step(time_step)
{
}

void VectorizedNetwork::addGridNode(TwoDLib::Mesh mesh, TwoDLib::TransitionMatrix tmat, double start_v, double start_w,
  std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res) {

  _num_nodes++;
  _node_id_to_group_mesh.insert(std::pair<MPILib::NodeId, MPILib::Index>(_num_nodes-1,_vec_mesh.size()));
  _group_mesh_to_node_id.insert(std::pair<MPILib::Index, MPILib::NodeId>(_vec_mesh.size(),_num_nodes-1));
  _grid_meshes.push_back(_vec_mesh.size());
  _vec_mesh.push_back(mesh);
  _vec_transforms.push_back(tmat);
  _start_vs.push_back(start_v);
  _start_ws.push_back(start_w);
  _vec_vec_rev.push_back(vec_rev);
  _vec_vec_res.push_back(vec_res);

}

void VectorizedNetwork::addMeshNode(TwoDLib::Mesh mesh,
  std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res) {

  _num_nodes++;
  _node_id_to_group_mesh.insert(std::pair<MPILib::NodeId, MPILib::Index>(_num_nodes-1,_vec_mesh.size()));
  _group_mesh_to_node_id.insert(std::pair<MPILib::Index, MPILib::NodeId>(_vec_mesh.size(),_num_nodes-1));
  _mesh_meshes.push_back(_vec_mesh.size());
  _vec_mesh.push_back(mesh);
  _vec_vec_rev.push_back(vec_rev);
  _vec_vec_res.push_back(vec_res);

}

void VectorizedNetwork::addRateNode(function_pointer functor){
  _num_nodes++;
  _rate_functions.push_back(function_association(_num_nodes-1,functor));
}

void VectorizedNetwork::initOde2DSystem(){

  _group = new TwoDLib::Ode2DSystemGroup(_vec_mesh,_vec_vec_rev,_vec_vec_res);

	for( MPILib::Index i=0; i < _grid_meshes.size(); i++){
    vector<TwoDLib::Coordinates> coords = _vec_mesh[i].findPointInMeshSlow(TwoDLib::Point(_start_vs[i], _start_ws[i]));
    _group->Initialize(i,coords[0][0],coords[0][1]);

    //setup initial working index space for each grid mesh
    std::set<unsigned int> working_index;
    working_index.insert(_group->Map(i, coords[0][0],coords[0][1]));
    _grid_working_index.push_back(working_index);

    //create CSR Matrix for each transforms
    _csrs.push_back(TwoDLib::CSRMatrix(_vec_transforms[i], *(_group), i));
  }

  // All grids/meshes must have the same timestep
  TwoDLib::MasterParameter par(static_cast<MPILib::Number>(ceil(_vec_mesh[0].TimeStep()/_network_time_step)));
  _n_steps = std::max((int)par._N_steps,10);

  _group_adapter = new CudaTwoDLib::CudaOde2DSystemAdapter(*(_group));
}

void VectorizedNetwork::rectifyWorkingIndexes(std::vector<inttype>& off1s, std::vector<inttype>& off2s){

  // calculate the largest noise spread
  inttype max_offset = 0;
  // find all cells which will be impacted by the master equation solver
  for(unsigned int o1=0; o1 < off1s.size(); o1++){
    max_offset = std::max((int)max_offset, (int)std::ceil((std::abs(off1s[o1]))));
  }
  for(unsigned int o2=0; o2 < off2s.size(); o2++){
    max_offset = std::max((int)max_offset, (int)std::ceil((std::abs(off2s[o2]))));
  }

  int noise_spread = max_offset*_n_steps;

  for (unsigned int m=0; m<_grid_meshes.size(); m++){
    unsigned int mesh_offset = _group->Offsets()[m];

    std::set<unsigned int> new_working_index_dynamics;
    std::set<unsigned int> new_working_index_dynamics_and_reset;

    std::set<unsigned int>::iterator it;
    for(it = _grid_working_index[m].begin(); it != _grid_working_index[m].end(); it++){
      unsigned int idx = *it;

      //eliminate all cells with mass less than epsilon
      if(_group->Mass()[idx] > 0.000001){
        new_working_index_dynamics.insert(idx);

        // find all cells which will recieve mass due to dynamics
        for(unsigned int t=0; t < _vec_transforms[m].Matrix()[idx-mesh_offset]._vec_to_line.size(); t++){
          unsigned int index = _group->Map(m,
            _vec_transforms[m].Matrix()[idx-mesh_offset]._vec_to_line[t]._to[0],
            _vec_transforms[m].Matrix()[idx-mesh_offset]._vec_to_line[t]._to[1]);
          new_working_index_dynamics.insert(index);
        }
      }
    }

    //add any possible reset cells
    for(it = new_working_index_dynamics.begin(); it != new_working_index_dynamics.end(); it++){
      unsigned int idx = *it;
      new_working_index_dynamics_and_reset.insert(idx);
      // ew - we have to go looking for the correct reset cells!
      for(int r=0; r<_vec_vec_res[m].size(); r++){
        if( _group->Map(m, _vec_vec_res[m][r]._from[0], _vec_vec_res[m][r]._from[1]) == idx){
          new_working_index_dynamics_and_reset.insert(_group->Map(m, _vec_vec_res[m][r]._to[0], _vec_vec_res[m][r]._to[1]));
        }
      }
    }

    _grid_working_index[m].clear();
    // expand the index to include all noise receiving cells
    int mesh_size = _group->Mass().size();
    for(it = new_working_index_dynamics_and_reset.begin(); it != new_working_index_dynamics_and_reset.end(); it++){
      int idx = *it;
      for(int n=-noise_spread; n<noise_spread; n++){
        int new_ind = ((((int)idx + (int)n)%(int)mesh_size)+(int)mesh_size)%(int)mesh_size;
        _grid_working_index[m].insert(new_ind);
      }
    }
  }

  // flatten the vector of sets into a single vector
  _grid_working_index_flattened = std::vector<inttype>(_group->Mass().size(),0);
  _grid_working_index_sizes.clear();
  for (unsigned int m=0; m< _grid_meshes.size(); m++){
    std::set<unsigned int>::iterator it;
    int i=0;
    for(it = _grid_working_index[m].begin(); it != _grid_working_index[m].end(); it++){
      unsigned int idx = *it;
      _grid_working_index_flattened[i+_group->Offsets()[m]] = idx;
      i++;
    }
    _grid_working_index_sizes.push_back(_grid_working_index[m].size());
  }
}

void VectorizedNetwork::reportNodeActivities(long sim_time){
  for (int i=0; i<_rate_nodes.size(); i++){
		std::ostringstream ost2;
		ost2 << "rate_" << i;
		std::ofstream ofst_rate(ost2.str(), std::ofstream::app);
		ofst_rate.precision(10);
		ofst_rate << sim_time << "\t" << _out_rates[i] << std::endl;
		ofst_rate.close();
	}
}

void VectorizedNetwork::addGridConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns){
  _grid_connections.push_back(NodeGridConnection(in,out,efficacy,n_conns));
}

void VectorizedNetwork::addMeshConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns, TwoDLib::TransitionMatrix *tmat){
  _mesh_connections.push_back(NodeMeshConnection(in,out,efficacy,n_conns,tmat));
}

void VectorizedNetwork::mainLoop(MPILib::Time t_begin, MPILib::Time t_end, MPILib::Time t_report, bool write_displays) {
	MPILib::Number n_iter = static_cast<MPILib::Number>(ceil((t_end - t_begin)/_network_time_step));
	MPILib::Number n_report = static_cast<MPILib::Number>(ceil((t_report - t_begin)/_network_time_step));

  for(unsigned int i=0; i<_display_nodes.size(); i++){
    TwoDLib::Display::getInstance()->addOdeSystem(i, _group, _node_id_to_group_mesh[i]);
  }

  const MPILib::Time h = 1./_n_steps*_vec_mesh[0].TimeStep();

  // Setup the OpenGL displays (if there are any required)
	// TwoDLib::Display::getInstance()->animate(write_displays, _display_nodes, _network_time_step);

  // Generate calculated transition vectors for grid derivative
  std::vector<inttype> node_to_group_meshes;
  std::vector<fptype> stays;
  std::vector<fptype> goes;
  std::vector<inttype> off1s;
  std::vector<inttype> off2s;

  for (unsigned int i=0; i<_grid_connections.size(); i++){
    // for each connection, which of group's meshes is being affected
    node_to_group_meshes.push_back(_node_id_to_group_mesh[_grid_connections[i]._out]);
    // the input rate comes from the node indexed by connection _in
    TwoDLib::Mesh::GridCellTransition cell_transition =
      _group->MeshObjects()[_node_id_to_group_mesh[_grid_connections[i]._out]]
      .calculateCellTransition(_grid_connections[i]._efficacy);
    stays.push_back(cell_transition._stays);
    goes.push_back(cell_transition._goes);
    off1s.push_back(cell_transition._offset_1);
    off2s.push_back(cell_transition._offset_2);
  }

  for (unsigned int i=0; i<_mesh_connections.size(); i++){
    node_to_group_meshes.push_back(_node_id_to_group_mesh[_mesh_connections[i]._out]);
    _csrs.push_back(TwoDLib::CSRMatrix(*(_mesh_connections[i]._transition), *(_group), _node_id_to_group_mesh[_mesh_connections[i]._out]));
  }

  CudaTwoDLib::CSRAdapter _csr_adapter(*_group_adapter,_csrs,_grid_meshes.size(),_grid_connections.size()+_mesh_connections.size(),h);

  MPILib::utilities::ProgressBar *pb = new MPILib::utilities::ProgressBar(n_iter);
	MPILib::Time time = 0;
  boost::timer::auto_cpu_timer timer;
	for(MPILib::Index i_loop = 0; i_loop < n_iter; i_loop++){
		time = _network_time_step*i_loop;

    // rectifyWorkingIndexes(off1s, off2s);
    // _group_adapter->TransferWorkingIndexData(_grid_working_index_flattened);

    std::vector<fptype> rates;
    for (unsigned int i=0; i<_grid_connections.size(); i++){
      rates.push_back(_out_rates[_grid_connections[i]._in]*_grid_connections[i]._n_connections);
    }
    for (unsigned int i=0; i<_mesh_connections.size(); i++){
      rates.push_back(_out_rates[_mesh_connections[i]._in]*_mesh_connections[i]._n_connections);
    }

		_group_adapter->Evolve(_mesh_meshes);

    // _csr_adapter.ClearDerivative();
		// _csr_adapter.SingleTransformStep(_grid_working_index_sizes);
    // _csr_adapter.AddDerivativeFull();

    _group_adapter->RedistributeProbability();
    // _group_adapter->MapFinish();

    // _group_adapter->updateGroupMass();

		for (MPILib::Index i_part = 0; i_part < _n_steps; i_part++ ){
			// _csr_adapter.ClearDerivative();
			// _csr_adapter.CalculateMeshGridDerivative(node_to_group_meshes, rates, stays, goes, off1s, off2s,_grid_working_index_sizes);
			// _csr_adapter.AddDerivative();
		}

    const std::vector<fptype>& group_rates = _group_adapter->F();
    for(unsigned int i=0; i<group_rates.size(); i++)
      _out_rates[_group_mesh_to_node_id[i]] = group_rates[i];

    for( const auto& element: _rate_functions)
  		_out_rates[element.first] = element.second(time);

    // _group_adapter->updateGroupWorkingIndex();
		// TwoDLib::Display::getInstance()->updateDisplay(i_loop);
		reportNodeActivities(time);

    (*pb)++;

	}
}
