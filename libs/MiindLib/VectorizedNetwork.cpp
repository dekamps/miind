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
  std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res, double tau_refractive) {

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
  _vec_tau_refractive.push_back(tau_refractive);

}

void VectorizedNetwork::addMeshNode(TwoDLib::Mesh mesh,
  std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res, double tau_refractive) {

  _num_nodes++;
  _node_id_to_group_mesh.insert(std::pair<MPILib::NodeId, MPILib::Index>(_num_nodes-1,_vec_mesh.size()));
  _group_mesh_to_node_id.insert(std::pair<MPILib::Index, MPILib::NodeId>(_vec_mesh.size(),_num_nodes-1));
  _mesh_meshes.push_back(_vec_mesh.size());
  _vec_mesh.push_back(mesh);
  _vec_vec_rev.push_back(vec_rev);
  _vec_vec_res.push_back(vec_res);
  _vec_tau_refractive.push_back(tau_refractive);

}

void VectorizedNetwork::addRateNode(function_pointer functor){
  _num_nodes++;
  _rate_functions.push_back(function_association(_num_nodes-1,functor));
}

void VectorizedNetwork::initOde2DSystem(unsigned int min_solve_steps){

  _group = new TwoDLib::Ode2DSystemGroup(_vec_mesh,_vec_vec_rev,_vec_vec_res,_vec_tau_refractive);

	for( MPILib::Index i=0; i < _grid_meshes.size(); i++){
    vector<TwoDLib::Coordinates> coords = _vec_mesh[_grid_meshes[i]].findPointInMeshSlow(TwoDLib::Point(_start_vs[i], _start_ws[i]));
    _group->Initialize(_grid_meshes[i],coords[0][0],coords[0][1]);

    //create CSR Matrix for each transforms
    _csrs.push_back(TwoDLib::CSRMatrix(_vec_transforms[i], *(_group), _grid_meshes[i]));
  }

  for(MPILib::Index i=0; i < _mesh_meshes.size(); i++){
    _group->Initialize(_mesh_meshes[i],0,0);
  }

  // All grids/meshes must have the same timestep
  TwoDLib::MasterParameter par(static_cast<MPILib::Number>(ceil(_network_time_step/_vec_mesh[0].TimeStep())));
  _n_steps = std::max((int)par._N_steps,(int)min_solve_steps);
  std::cout << "Using master solver n_steps = " << _n_steps << "\n";

  _group_adapter = new CudaTwoDLib::CudaOde2DSystemAdapter(*(_group));
}

void VectorizedNetwork::reportNodeActivities(MPILib::Time sim_time){
  for (int i=0; i<_rate_nodes.size(); i++){
		std::ostringstream ost2;
		ost2 << "rate_" << i;
		std::ofstream ofst_rate(ost2.str(), std::ofstream::app);
		ofst_rate.precision(10);
		ofst_rate << sim_time << "\t" << _out_rates[i] << std::endl;
		ofst_rate.close();
	}
}

void VectorizedNetwork::reportNodeDensities(MPILib::Time sim_time){
  for (int i=0; i<_density_nodes.size(); i++){
    if(sim_time < _density_start_times[i] || sim_time > _density_end_times[i] || std::fabs(std::remainder(sim_time, _density_intervals[i])) > 0.00000001 )
      continue;

    std::ostringstream ost;
    ost << _density_nodes[i]  << "_" << sim_time;
    string fn("node_" + ost.str());

    std::string model_path("densities");
    boost::filesystem::path path(model_path);

    // MdK 27/01/2017. grid file is now created in the cwd of the program and
    // not in the directory where the mesh resides.
    const std::string dirname = path.filename().string();

    if (! boost::filesystem::exists(dirname) ){
      boost::filesystem::create_directory(dirname);
    }
    std::ofstream ofst(dirname + "/" + fn);
    _group->DumpSingleMesh(&ofst, _density_nodes[i]);
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
  if(_display_nodes.size() > 0){
	   TwoDLib::Display::getInstance()->animate(write_displays, _display_nodes, _network_time_step);
   }

  // Generate calculated transition vectors for grid derivative
  std::vector<inttype> node_to_group_meshes;
  std::vector<fptype> stays;
  std::vector<fptype> goes;
  std::vector<int> off1s;
  std::vector<int> off2s;

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

    std::vector<fptype> rates;
    for (unsigned int i=0; i<_grid_connections.size(); i++){
      rates.push_back(_out_rates[_grid_connections[i]._in]*_grid_connections[i]._n_connections);
    }
    for (unsigned int i=0; i<_mesh_connections.size(); i++){
      rates.push_back(_out_rates[_mesh_connections[i]._in]*_mesh_connections[i]._n_connections);
    }

		_group_adapter->Evolve(_mesh_meshes);

    _csr_adapter.ClearDerivative();
    _csr_adapter.SingleTransformStep();
    _csr_adapter.AddDerivativeFull();

    _group_adapter->RedistributeProbability();
    _group_adapter->MapFinish();

		for (MPILib::Index i_part = 0; i_part < _n_steps; i_part++ ){
			_csr_adapter.ClearDerivative();
      _csr_adapter.CalculateMeshGridDerivative(node_to_group_meshes,rates,stays, goes, off1s, off2s);
			_csr_adapter.AddDerivative();
		}

    const std::vector<fptype>& group_rates = _group_adapter->F();
    for(unsigned int i=0; i<group_rates.size(); i++){
      _out_rates[_group_mesh_to_node_id[i]] = group_rates[i];
    }

    for( const auto& element: _rate_functions){
  		_out_rates[element.first] = element.second(time);
    }

    if(_display_nodes.size() > 0){
      _group_adapter->updateGroupMass();
      TwoDLib::Display::getInstance()->updateDisplay(i_loop);
    }

    if(_density_nodes.size() > 0){
      _group_adapter->updateGroupMass();
      reportNodeDensities(time);
    }

		reportNodeActivities(time);

    (*pb)++;

	}
}
