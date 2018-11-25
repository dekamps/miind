#include "VectorizedNetwork.hpp"

using namespace MiindLib;

VectorizedNetwork::VectorizedNetwork(MPILib::Time time_step):
_num_nodes(0),
_n_steps(10),
_network_time_step(time_step)
{
}

VectorizedNetwork::~VectorizedNetwork(){
}

void VectorizedNetwork::addGridNode(TwoDLib::Mesh mesh, TwoDLib::TransitionMatrix tmat, double start_v, double start_w,
  std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res) {

  _num_nodes++;
  _node_id_to_grid_mesh.insert(std::pair<MPILib::NodeId, MPILib::Index>(_num_nodes-1,_vec_mesh.size()));
  _grid_mesh_to_node_id.insert(std::pair<MPILib::Index, MPILib::NodeId>(_vec_mesh.size(),_num_nodes-1));
  _vec_mesh.push_back(mesh);
  _vec_transforms.push_back(tmat);
  _start_vs.push_back(start_v);
  _start_ws.push_back(start_w);
  _vec_vec_rev.push_back(vec_rev);
  _vec_vec_res.push_back(vec_res);

}

void VectorizedNetwork::addRateNode(function_pointer functor){
  _num_nodes++;
  _rate_functions.push_back(function_association(_num_nodes-1,functor));
}

void VectorizedNetwork::initOde2DSystem(){

  _group = new TwoDLib::Ode2DSystemGroup(_vec_mesh,_vec_vec_rev,_vec_vec_res);

	for( MPILib::Index i=0; i < _vec_mesh.size(); i++){
    _num_nodes++;
    vector<TwoDLib::Coordinates> coords = _vec_mesh[i].findPointInMeshSlow(TwoDLib::Point(_start_vs[i], _start_ws[i]));
    _group->Initialize(i,coords[0][0],coords[0][1]);
  }

  for( MPILib::Index i=0; i < _vec_transforms.size(); i++)
    _csrs.push_back(TwoDLib::CSRMatrix(_vec_transforms[i], *(_group), i));

  // All grids/meshes must have the same timestep
  TwoDLib::MasterParameter par(static_cast<MPILib::Number>(ceil(_vec_mesh[0].TimeStep()/_network_time_step)));
  _n_steps = par._N_steps * 10;

  _group_adapter = new CudaTwoDLib::CudaOde2DSystemAdapter(*(_group));
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

void VectorizedNetwork::addConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns){
  _connections.push_back(NodeConnection(in,out,efficacy,n_conns));
}

void VectorizedNetwork::mainLoop(MPILib::Time t_begin, MPILib::Time t_end, MPILib::Time t_report, bool write_displays) {
	MPILib::Number n_iter = static_cast<MPILib::Number>(ceil((t_end - t_begin)/_network_time_step));
	MPILib::Number n_report = static_cast<MPILib::Number>(ceil((t_report - t_begin)/_network_time_step));

  for(unsigned int i=0; i<_display_nodes.size(); i++){
    TwoDLib::Display::getInstance()->addOdeSystem(0, _group, _node_id_to_grid_mesh[i]);
  }


  const MPILib::Time h = 1./_n_steps*_vec_mesh[0].TimeStep();

	CudaTwoDLib::CSRAdapter _csr_adapter(*_group_adapter,_csrs,_connections.size(),h);

  // Setup the OpenGL displays (if there are any required)
	TwoDLib::Display::getInstance()->animate(write_displays, _display_nodes, _network_time_step);

  // Generate calculated transition vectors for grid derivative
  std::vector<inttype> node_to_group_meshes;
  std::vector<fptype> stays;
  std::vector<fptype> goes;
  std::vector<inttype> off1s;
  std::vector<inttype> off2s;

  for (unsigned int i=0; i<_connections.size(); i++){
    // for each connection, which of group's meshes is being affected
    node_to_group_meshes.push_back(_node_id_to_grid_mesh[_connections[i]._out]);
    // the input rate comes from the node indexed by connection _in
    TwoDLib::Mesh::GridCellTransition cell_transition =
      _group->MeshObjects()[_node_id_to_grid_mesh[_connections[i]._out]]
      .calculateCellTransition(_connections[i]._efficacy);
    stays.push_back(cell_transition._stays);
    goes.push_back(cell_transition._goes);
    off1s.push_back(cell_transition._offset_1);
    off2s.push_back(cell_transition._offset_2);
  }

	MPILib::Time time = 0;
	for(MPILib::Index i_loop = 0; i_loop < n_iter; i_loop++){
		time = _network_time_step*i_loop;

    std::vector<fptype> rates;
    for (unsigned int i=0; i<_connections.size(); i++){
      rates.push_back(_out_rates[_connections[i]._in]*_connections[i]._n_connections);
    }

		_group_adapter->EvolveWithoutMeshUpdate();
		_csr_adapter.SingleTransformStep();

    // _group_adapter->RedistributeProbability();
    // _group_adapter->MapFinish();

    _group_adapter->updateGroupMass();

		for (MPILib::Index i_part = 0; i_part < _n_steps; i_part++ ){
			_csr_adapter.ClearDerivative();
			_csr_adapter.CalculateGridDerivative(node_to_group_meshes, rates, stays, goes, off1s, off2s);
			_csr_adapter.AddDerivative();
		}

    const std::vector<fptype>& group_rates = _group_adapter->F();
    for(unsigned int i=0; i<group_rates.size(); i++)
      _out_rates[_grid_mesh_to_node_id[i]] = group_rates[i];

    for( const auto& element: _rate_functions)
  		_out_rates[element.first] = element.second(time);

		TwoDLib::Display::getInstance()->updateDisplay(i_loop);
		reportNodeActivities(time);

	}
}
