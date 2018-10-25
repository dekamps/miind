#ifndef _CODE_LIBS_TWODLIBLIB_GRIDALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIBLIB_GRIDALGORITHMCODE_INCLUDE_GUARD

#include <boost/filesystem.hpp>
#include <MPILib/include/utilities/Log.hpp>
#include <fstream>
#include <iostream>
#include "GridAlgorithm.hpp"
#include "Stat.hpp"
#include "TwoDLibException.hpp"
#include "display.hpp"

namespace TwoDLib {

	template <class WeightValue>
	GridAlgorithm<WeightValue>::GridAlgorithm
	(
		const std::string& model_name,
		const std::string& transform_matrix,
		MPILib::Time h,
		MPILib::Index start_strip,
		MPILib::Index start_cell,
		MPILib::Time tau_refractive,
		const std::string&  rate_method
	):
	_model_name(model_name),
	_rate_method(rate_method),
	_rate(0.0),
	_t_cur(0.0),
	_root(CreateRootNode(model_name)),
	_mesh(CreateMeshObject()),
	_vec_rev(Mapping("Reversal")),
	_vec_res(Mapping("Reset")),
	_dt(_mesh.TimeStep()),
	_sys(_mesh,_vec_rev,_vec_res,tau_refractive),
	_n_evolve(0),
	_n_steps(0),
	_sysfunction(rate_method == "AvgV" ? &TwoDLib::Ode2DSystem::AvgV : &TwoDLib::Ode2DSystem::F),
	_start_strip(start_strip),
	_start_cell(start_cell),
	_transform_matrix(transform_matrix)
	{
		_mass_swap = vector<double>(_sys._vec_mass.size());

		// default initialization is (0,0); if there is no strip 0, it's down to the user
		_sys.Initialize(_start_strip,_start_cell);

	}

	template <class WeightValue>
	GridAlgorithm<WeightValue>::GridAlgorithm(const GridAlgorithm<WeightValue>& rhs):
	_model_name(rhs._model_name),
	_rate_method(rhs._rate_method),
	_rate(rhs._rate),
	_t_cur(rhs._t_cur),
	_mesh(rhs._mesh),
	_root(rhs._root),
	_vec_rev(rhs._vec_rev),
	_vec_res(rhs._vec_res),
	_dt(_mesh.TimeStep()),
	_sys(_mesh,_vec_rev,_vec_res,rhs._sys.Tau_ref()),
	_n_evolve(0),
	_n_steps(0),
	_sysfunction(rhs._sysfunction),
	_start_strip(rhs._start_strip),
	_start_cell(rhs._start_cell),
	_transform_matrix(rhs._transform_matrix)
	{
		_mass_swap = vector<double>(_sys._vec_mass.size());

		_sys.Initialize(_start_strip,_start_cell);

		Display::getInstance()->addOdeSystem(&_sys);
	}

  template <class WeightValue>
	GridAlgorithm<WeightValue>* GridAlgorithm<WeightValue>::clone() const
	{
	  return new GridAlgorithm<WeightValue>(*this);
	}

	template <class WeightValue>
	void GridAlgorithm<WeightValue>::configure(const MPILib::SimulationRunParameter& par_run)
	{
		_transformMatrix = TransitionMatrix(_transform_matrix);
		_csr_transform = new CSRMatrix(_transformMatrix, _sys);

		//Display::getInstance()->addOdeSystem(&_sys);

		_t_cur = par_run.getTBegin();
		MPILib::Time t_step     = par_run.getTStep();

		Quadrilateral q = _sys.MeshObject().Quad(1,0);
		std::vector<Point> ps = q.Points();

		double cell_min = q.Centroid()[0];
		double cell_max = q.Centroid()[0];

		for (int p=0; p<ps.size(); p++){
			if (ps[p][0] < cell_min)
				cell_min = ps[p][0];
			if (ps[p][0] > cell_max)
				cell_max = ps[p][0];
		}

		double cell_width = cell_max - cell_min; //check the width of cell 1,0 - they should all be the same!

		try {
			std::unique_ptr<MasterGrid> p_master(new MasterGrid(_sys,cell_width,101));
			_p_master = std::move(p_master);
		}
		// TODO: investigate the following
		// for some reason, the exception is usually not caught by the main program, which is why we write its message to cerr here.
		catch(TwoDLibException& e){
			std::cerr << e.what() << std::endl;
			throw e;
		}

		// at this stage initialization must have taken place, either by default in (0,0),
		// or by the user calling Initialize if there is no strip 0

		double sum = _sys.P();
		if (sum == 0.)
			throw TwoDLib::TwoDLibException("No initialization of the mass array has taken place. Call Initialize before configure.");

	}

	template <class WeightValue>
	std::vector<TwoDLib::Redistribution> GridAlgorithm<WeightValue>::Mapping(const string& type)
	{
		Pred pred(type);
		pugi::xml_node rev_node = _root.find_child(pred);

		if (rev_node.name() != std::string("Mapping") ||
		    rev_node.attribute("type").value() != type)
			throw TwoDLibException("Couldn't find mapping in model file");

		std::ostringstream ostrev;
		rev_node.print(ostrev);
		std::istringstream istrev(ostrev.str());
		vector<TwoDLib::Redistribution> vec_rev = TwoDLib::ReMapping(istrev);
		return vec_rev;
	}

	template <class WeightValue>
	Mesh GridAlgorithm<WeightValue>::CreateMeshObject(){
		// mesh
		pugi::xml_node mesh_node = _root.first_child();

		if (mesh_node.name() != std::string("Mesh") )
		  throw TwoDLib::TwoDLibException("Couldn't find mesh node in model file");
		std::ostringstream ostmesh;
		mesh_node.print(ostmesh);
		std::istringstream istmesh(ostmesh.str());

		TwoDLib::Mesh mesh(istmesh);

		// MatrixGenerator should already have inserted the stationary bin and there is no need
		// to reexamine the stat file

		return mesh;
	}

	template <class WeightValue>
	pugi::xml_node GridAlgorithm<WeightValue>::CreateRootNode(const string& model_name){

		// document
		pugi::xml_parse_result result = _doc.load_file(model_name.c_str());
		pugi::xml_node  root = _doc.first_child();

		if (result.status != pugi::status_ok)
		  throw TwoDLib::TwoDLibException("Can't open .model file.");
		return root;
	}

	template <class WeightValue>
	void GridAlgorithm<WeightValue>::evolveNodeState
	(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
		MPILib::Time time,
		const std::vector<MPILib::NodeType>& typeVector
	)
	{
	  // The network time step must be an integer multiple of the network time step; in principle
	  // we would expect this multiple to be one, but perhaps there are reasons to allow a population
	  // have a finer time resolution than others, so we allow larger multiples but write a warning in the log file.
		// determine number of steps and fix at the first step.
		if (_n_steps == 0){
		  // since n_steps == 0, time is the network time step
			double n = (time - _t_cur)/_dt;

			_n_steps = static_cast<MPILib::Number>(round(n));
			if (_n_steps == 0){

			  throw TwoDLibException("Network time step is smaller than this grid's time step.");
			}
			if (fabs(_n_steps - n) > 1e-6){
			  throw TwoDLibException("Mismatch of mesh time step and network time step. Network time step should be a multiple (mostly one) of network time step");
			}
			if (_n_steps > 1)
			  LOG(MPILib::utilities::logWARNING) << "Mesh runs at a time step which is a multiple of the network time step. Is this intended?";
			else
			  ; // else is fine
		}

			Display::getInstance()->LockMutex();
	    // mass rotation
	    for (MPILib::Index i = 0; i < _n_steps; i++){

				_sys.EvolveWithoutMeshUpdate();
#pragma omp parallel for
				 for(unsigned int id = 0; id < _mass_swap.size(); id++)
					  _mass_swap[id] = 0.;

			 	_csr_transform->MV(_mass_swap,_sys._vec_mass);

				_sys._vec_mass = _mass_swap;
	    }

	    // master equation
	    _p_master->Apply(_n_steps*_dt,_vec_rates,_efficacy_map);

	    _sys.RedistributeProbability();

 	    _t_cur += _n_steps*_dt;
 	    _rate = (_sys.*_sysfunction)();

 	    _n_evolve++;

			Display::getInstance()->UnlockMutex();

			//Display::getInstance()->updateDisplay();
	}

	template <class WeightValue>
	void GridAlgorithm<WeightValue>::FillMap(const std::vector<WeightValue>& vec_weights)
	{
		// this function will only be called once;
		_efficacy_map = std::vector<double>(vec_weights.size());

 		for(MPILib::Index i_weight = 0; i_weight < _efficacy_map.size(); i_weight++){
			_efficacy_map[i_weight] = vec_weights[i_weight]._efficacy;
		}

 		_vec_rates = std::vector<double>(vec_weights.size(),0.);
	}

	template <class WeightValue>
	MPILib::AlgorithmGrid GridAlgorithm<WeightValue>::getGrid(MPILib::NodeId id, bool b_state) const
	{
		// An empty grid will lead to crashes
		vector<double> array_interpretation {0.};
		vector<double> array_state {0.};

		if (b_state){
			// Output to a rate file as well. This might be slow, but we can observe
			// the rate as the simulation progresses rather than wait for root.
			std::ostringstream ost2;
			ost2 << "rate_" << id ;
			std::ofstream ofst_rate(ost2.str(), std::ofstream::app);
			ofst_rate.precision(10);
			ofst_rate << _t_cur << "\t" << _sys.F() << std::endl;
			ofst_rate.close();
		}
		return MPILib::AlgorithmGrid(array_state,array_interpretation);
	}

	template <class WeightValue>
	void GridAlgorithm<WeightValue>::prepareEvolve
	(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
		const std::vector<MPILib::NodeType>& typeVector
	)
	{
		if (_efficacy_map.size() == 0)
			FillMap(weightVector);
		// take into account the number of connections

		assert(nodeVector.size() == weightVector.size());
		for (MPILib::Index i = 0; i < nodeVector.size(); i++)
			_vec_rates[i] = nodeVector[i]*weightVector[i]._number_of_connections;
	}


}

#endif
