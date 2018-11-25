#ifndef _CODE_CUDATWODLIB_VectorizedNetwork_INCLUDE_GUARD
#define _CODE_CUDATWODLIB_VectorizedNetwork_INCLUDE_GUARD

#include <CudaTwoDLib/CudaTwoDLib.hpp>

typedef CudaTwoDLib::fptype fptype;
typedef CudaTwoDLib::inttype inttype;
typedef MPILib::Rate (*function_pointer)(MPILib::Time);
typedef std::pair<MPILib::Index, function_pointer> function_association;
typedef std::vector<function_association> function_list;

namespace MiindLib {

class NodeConnection {
public:
  MPILib::NodeId _in;
  MPILib::NodeId _out;
  double _efficacy;
  int _n_connections;

  NodeConnection(MPILib::NodeId in, MPILib::NodeId out, double eff, int n_conns):
  _in(in),_out(out),_efficacy(eff),_n_connections(n_conns){}
};

class VectorizedNetwork {
public:
  VectorizedNetwork(MPILib::Time time_step);

  ~VectorizedNetwork();

  void initOde2DSystem();

  void setRateNodes(std::vector<MPILib::NodeId> ids){
    _rate_nodes = ids;
  }
  void setDisplayNodes(std::vector<MPILib::NodeId> ids){
    _display_nodes = ids;
  }
  void setDensityNodes(std::vector<MPILib::NodeId> ids){
    _density_nodes = ids;
  }

  void addGridNode(TwoDLib::Mesh mesh, TwoDLib::TransitionMatrix tmat, double start_v, double start_w,
    std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res);

  void addRateNode(function_pointer functor);

  void addConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns);

  void reportNodeActivities(long sim_time);
  void mainLoop(MPILib::Time t_begin, MPILib::Time t_end, MPILib::Time t_report, bool write_displays);

protected:

  std::vector<TwoDLib::TransitionMatrix> _vec_transforms;
  std::vector<double> _start_vs;
  std::vector<double> _start_ws;
  std::vector<TwoDLib::Mesh> _vec_mesh;
  std::vector< std::vector<TwoDLib::Redistribution> > _vec_vec_rev;
  std::vector< std::vector<TwoDLib::Redistribution> > _vec_vec_res;

  TwoDLib::Ode2DSystemGroup *_group;

  CudaTwoDLib::CudaOde2DSystemAdapter *_group_adapter;

  std::vector<TwoDLib::CSRMatrix> _csrs;

  unsigned int _n_steps;

  unsigned int _num_nodes;
  MPILib::Time _network_time_step;

  std::vector<MPILib::NodeId> _display_nodes;
  std::vector<MPILib::NodeId> _rate_nodes;
  std::vector<MPILib::NodeId> _density_nodes;

  std::vector<NodeConnection> _connections;

  std::map<MPILib::NodeId, double> _out_rates;

  function_list _rate_functions;

  std::map<MPILib::NodeId, MPILib::Index> _node_id_to_grid_mesh;
  std::map<MPILib::Index, MPILib::NodeId> _grid_mesh_to_node_id;

  std::map<MPILib::NodeId, TwoDLib::CSRMatrix> _transforms;

};

}

#endif
