#ifndef _CODE_MIINDLIB_VectorizedNetwork_INCLUDE_GUARD
#define _CODE_MIINDLIB_VectorizedNetwork_INCLUDE_GUARD

#include <CudaTwoDLib/CudaTwoDLib.hpp>

typedef CudaTwoDLib::fptype fptype;
typedef CudaTwoDLib::inttype inttype;
typedef MPILib::Rate (*function_pointer)(MPILib::Time);
typedef std::pair<MPILib::Index, function_pointer> function_association;
typedef std::vector<function_association> function_list;

namespace MiindLib {

class NodeMeshConnection {
public:
  MPILib::NodeId _in;
  MPILib::NodeId _out;
  double _efficacy;
  TwoDLib::TransitionMatrix *_transition;
  int _n_connections;

  NodeMeshConnection(MPILib::NodeId in, MPILib::NodeId out, double eff, int n_conns, TwoDLib::TransitionMatrix *trans):
  _in(in),_out(out),_efficacy(eff),_n_connections(n_conns), _transition(trans){}
};

class NodeGridConnection {
public:
  MPILib::NodeId _in;
  MPILib::NodeId _out;
  double _efficacy;
  int _n_connections;

  NodeGridConnection(MPILib::NodeId in, MPILib::NodeId out, double eff, int n_conns):
  _in(in),_out(out),_efficacy(eff),_n_connections(n_conns){}
};

class VectorizedNetwork {
public:
  VectorizedNetwork(MPILib::Time time_step);

  void initOde2DSystem(unsigned int min_solve_steps=10);

  void setRateNodes(std::vector<MPILib::NodeId> ids){
    _rate_nodes = ids;
  }
  void setDisplayNodes(std::vector<MPILib::NodeId> ids){
    _display_nodes = ids;
  }
  void setDensityNodes(std::vector<MPILib::NodeId> ids, std::vector<MPILib::Time> start_times,
    std::vector<MPILib::Time> end_times, std::vector<MPILib::Time> intervals){
    _density_nodes = ids;
  }

  void addGridNode(TwoDLib::Mesh mesh, TwoDLib::TransitionMatrix tmat, double start_v, double start_w,
    std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res, double tau_refractive);

  void addMeshNode(TwoDLib::Mesh mesh, std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res, double tau_refractive);

  void addRateNode(function_pointer functor);

  void addGridConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns);

  void addMeshConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns, TwoDLib::TransitionMatrix *tmat);

  void reportNodeActivities(MPILib::Time sim_time);
  void mainLoop(MPILib::Time t_begin, MPILib::Time t_end, MPILib::Time t_report, bool write_displays);

protected:

  std::vector<TwoDLib::TransitionMatrix> _vec_transforms;
  std::vector<double> _start_vs;
  std::vector<double> _start_ws;
  std::vector<TwoDLib::Mesh> _vec_mesh;
  std::vector< std::vector<TwoDLib::Redistribution> > _vec_vec_rev;
  std::vector< std::vector<TwoDLib::Redistribution> > _vec_vec_res;
  std::vector<MPILib::Time> _vec_tau_refractive;

  std::vector<inttype> _grid_meshes;
  std::vector<inttype> _mesh_meshes;

  TwoDLib::Ode2DSystemGroup *_group;

  CudaTwoDLib::CudaOde2DSystemAdapter *_group_adapter;

  std::vector<TwoDLib::CSRMatrix> _csrs;

  unsigned int _n_steps;

  unsigned int _num_nodes;
  MPILib::Time _network_time_step;

  std::vector<MPILib::NodeId> _display_nodes;
  std::vector<MPILib::NodeId> _rate_nodes;
  std::vector<MPILib::NodeId> _density_nodes;
  std::vector<MPILib::Time> _density_start_times;
  std::vector<MPILib::Time> _density_end_times;
  std::vector<MPILib::Time> _density_intervals;

  std::vector<NodeMeshConnection> _mesh_connections;
  std::vector<NodeGridConnection> _grid_connections;

  std::map<MPILib::NodeId, double> _out_rates;

  function_list _rate_functions;

  std::map<MPILib::NodeId, MPILib::Index> _node_id_to_group_mesh;
  std::map<MPILib::Index, MPILib::NodeId> _group_mesh_to_node_id;

};

}

#endif
