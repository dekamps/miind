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

	template <class WeightValue, class Solver>
	GridAlgorithm<WeightValue,Solver>::GridAlgorithm
	(
		const std::string& model_name,
		const std::string& transform_matrix,
		MPILib::Time h,
		MPILib::Time tau_refractive,
		const std::string&  rate_method,
		MPILib::Index start_strip,
		MPILib::Index start_cell
	):MeshAlgorithm<WeightValue,Solver>(model_name, std::vector<std::string>() ,h,tau_refractive,rate_method),
	_start_strip(start_strip),
	_start_cell(start_cell),
	_transform_matrix(transform_matrix)
	{
		_mass_swap = vector<double>(_sys._vec_mass.size());
		// default initialization is (0,0); if there is no strip 0, it's down to the user
		_sys.Initialize(_start_strip,_start_cell);

		Display::getInstance()->addOdeSystem(&_sys);
	}

	template <class WeightValue, class Solver>
	GridAlgorithm<WeightValue,Solver>::GridAlgorithm(const GridAlgorithm<WeightValue,Solver>& rhs):
  MeshAlgorithm<WeightValue,Solver>(rhs)
	{}

  template <class WeightValue,class Solver>
	GridAlgorithm<WeightValue,Solver>* GridAlgorithm<WeightValue,Solver>::clone() const
	{
	  return new GridAlgorithm<WeightValue,Solver>(*this);
	}

	template <class WeightValue, class Solver>
	void GridAlgorithm<WeightValue,Solver>::configure(const MPILib::SimulationRunParameter& par_run)
	{
		_transformMatrix = TransitionMatrix(_transform_matrix);
		_csr_transform = new CSRMatrix(_transformMatrix, _sys);

		_t_cur = par_run.getTBegin();
		MPILib::Time t_step     = par_run.getTStep();

		// the integration time step, stored in the MasterParameter, is gauged with respect to the
		// network time step.
		MPILib::Number n_ode = static_cast<MPILib::Number>(std::floor(t_step/_h));
		MasterParameter par((n_ode > 1) ? n_ode : 1);

		// vec_mat will go out of scope; MasterOMP will convert the matrices
		// internally and we don't want to keep two versions.
		std::vector<TransitionMatrix> vec_mat = InitializeMatrices(_mat_names);

		try {
			std::unique_ptr<Solver> p_master(new Solver(_sys,vec_mat, par));
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

	template <class WeightValue, class Solver>
	void GridAlgorithm<WeightValue,Solver>::evolveNodeState
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

	    // mass rotation
	    for (MPILib::Index i = 0; i < _n_steps; i++){
				_sys.Evolve();
#pragma omp parallel for
				 for(unsigned int id = 0; id < _mass_swap.size(); id++)
					 _mass_swap[id] = 0.;

	      _csr_transform->MV(_mass_swap,_sys._vec_mass);
				_sys._vec_mass = _mass_swap;
	    }

	    // master equation
	    _p_master->Apply(_n_steps*_dt,_vec_rates,_vec_map);

	    _sys.RedistributeProbability();

 	    _t_cur += _n_steps*_dt;
 	    _rate = (_sys.*_sysfunction)();

 	    _n_evolve++;
	}

	template <class WeightValue, class Solver>
	void GridAlgorithm<WeightValue,Solver>::FillMap(const std::vector<WeightValue>& vec_weights)
	{
 		_vec_rates = std::vector<double>(vec_weights.size(),0.);
	}


}

#endif
