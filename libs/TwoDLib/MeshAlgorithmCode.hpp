// Copyright (c) 2005 - 2015 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _CODE_LIBS_TWODLIBLIB_MESHALGORITHMCODE_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIBLIB_MESHALGORITHMCODE_INCLUDE_GUARD

#include <boost/filesystem.hpp>
#include <MPILib/include/utilities/Log.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include "MeshAlgorithm.hpp"
#include "Stat.hpp"
#include "TwoDLibException.hpp"
#include "display.hpp"
#include "GridReport.hpp"

namespace {

	// predicate that helps to locate "Mapping" nodes in an xml structure
	class Pred {
	public:
		Pred(const std::string& type):_type(type){}

		bool operator() (pugi::xml_node node){

			return (std::string(node.name()) == "Mapping" && std::string(node.attribute("type").value()) == _type ) ? true : false;
		}

	private:
		std::string _type;
	};
}

namespace TwoDLib {

	template <class WeightValue,class Solver>
	pugi::xml_node MeshAlgorithm<WeightValue,Solver>::CreateRootNode(const string& model_name){

		// document
		pugi::xml_parse_result result = _doc.load_file(model_name.c_str());
		pugi::xml_node  root = _doc.first_child();

		if (result.status != pugi::status_ok)
		  throw TwoDLib::TwoDLibException("Can't open .model file.");
		return root;
	}

	template <class WeightValue, class Solver>
	std::vector<TwoDLib::Mesh> MeshAlgorithm<WeightValue,Solver>::CreateMeshObject(){
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
 		std::vector<TwoDLib::Mesh> vec_mesh{ mesh };
		return vec_mesh;
	}


	template <class WeightValue, class Solver>
	std::vector<TwoDLib::Redistribution> MeshAlgorithm<WeightValue,Solver>::Mapping(const string& type)
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

	template <class WeightValue, class Solver>
	MeshAlgorithm<WeightValue,Solver>::MeshAlgorithm
	(
		const std::string& model_name,
		const std::vector<std::string>& mat_names,
		MPILib::Time h,
		MPILib::Time tau_refractive,
		const std::string&  rate_method
	):
	_tolerance(1e-7),
	_model_name(model_name),
	_mat_names(mat_names),
	_rate_method(rate_method),
	_h(h),
	_rate(0.0),
	_t_cur(0.0),
	_root(CreateRootNode(model_name)),
	_mesh_vec(CreateMeshObject()),
	_vec_vec_rev(std::vector<std::vector<Redistribution> >{this->Mapping("Reversal")}),
	_vec_vec_res(std::vector<std::vector<Redistribution> >{this->Mapping("Reset")}),
	_vec_tau_refractive(std::vector<MPILib::Time>({tau_refractive})),
	_vec_map(0),
	_dt(_mesh_vec[0].TimeStep()),
	_sys(_mesh_vec,_vec_vec_rev,_vec_vec_res,_vec_tau_refractive),
	_n_evolve(0),
	_n_steps(0),
	// AvgV method is for Fitzhugh-Nagumo, and other methods that don't have a threshold crossing
	_sysfunction(rate_method == "AvgV" ? &TwoDLib::Ode2DSystemGroup::AvgV : &TwoDLib::Ode2DSystemGroup::F)
	// master parameter can only be calculated on configuration
	{
		// default initialization is (0,0); if there is no strip 0, it's down to the user
		if (_mesh_vec[0].NrCellsInStrip(0) > 0 )
			_sys.Initialize(0,0,0);
	}

	template <class WeightValue, class Solver>
	MeshAlgorithm<WeightValue,Solver>::MeshAlgorithm(const MeshAlgorithm<WeightValue,Solver>& rhs):
	_tolerance(rhs._tolerance),
	_model_name(rhs._model_name),
	_mat_names(rhs._mat_names),
	_rate_method(rhs._rate_method),
	_h(rhs._h),
	_rate(rhs._rate),
	_t_cur(rhs._t_cur),
	_mesh_vec(rhs._mesh_vec),
	_vec_vec_rev(rhs._vec_vec_rev),
	_vec_vec_res(rhs._vec_vec_res),
	_vec_tau_refractive(rhs._vec_tau_refractive),
	_vec_map(0),
	_dt(_mesh_vec[0].TimeStep()),
	_sys(_mesh_vec,_vec_vec_rev,_vec_vec_res,_vec_tau_refractive),
	_n_evolve(0),
	_n_steps(0),
	_sysfunction(rhs._sysfunction)
	// master parameter can only be calculated on configuration
	{
		// default initialization is (0,0); if there is no strip 0, it's down to the user
		if (_mesh_vec[0].NrCellsInStrip(0) > 0 )
			_sys.Initialize(0,0,0);
	}

	template <class WeightValue, class Solver>
	std::vector<TwoDLib::TransitionMatrix> MeshAlgorithm<WeightValue,Solver>::InitializeMatrices(const std::vector<std::string>& mat_names)
	{
		std::vector<TwoDLib::TransitionMatrix> vec_mat;

		for (const auto& name: mat_names)
			vec_mat.push_back(TransitionMatrix(name));

		return vec_mat;
	}

	template <class WeightValue, class Solver>
	MeshAlgorithm<WeightValue,Solver>* MeshAlgorithm<WeightValue,Solver>::clone() const
	{
	  return new MeshAlgorithm<WeightValue,Solver>(*this);
	}

	template <class WeightValue, class Solver>
	void MeshAlgorithm<WeightValue,Solver>::configure(const MPILib::SimulationRunParameter& par_run)
	{
		_t_cur = par_run.getTBegin();
		MPILib::Time t_step     = par_run.getTStep();

		Display::getInstance()->addOdeSystem(_node_id, &_sys);
		GridReport<WeightValue>::getInstance()->registerObject(_node_id, this);

		// the integration time step, stored in the MasterParameter, is gauged with respect to the
		// network time step.
		MPILib::Number n_ode = static_cast<MPILib::Number>(std::floor(t_step/_h));
		MasterParameter par(100);

		// vec_mat will go out of scope; MasterOMP will convert the matrices
		// internally and we don't want to keep two versions.
		std::vector<TransitionMatrix> vec_mat = InitializeMatrices(_mat_names);

		try {
			std::unique_ptr<Solver> p_master(new Solver(_sys,std::vector<std::vector<TransitionMatrix > >{vec_mat}, par));
			_p_master = std::move(p_master);
		}
		// TODO: investigate the following
		// for some reason, the rethrown exception is usually not caught by the main program, which is why we write its message to cerr here.
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
	void MeshAlgorithm<WeightValue,Solver>::assignNodeId( MPILib::NodeId nid ) {
		_node_id = nid;
	}

	template <class WeightValue,class Solver>
	MPILib::AlgorithmGrid MeshAlgorithm<WeightValue,Solver>::getGrid(MPILib::NodeId id, bool b_state) const
	{
		// An empty grid will lead to crashes
		vector<double> array_interpretation {0.};
		vector<double> array_state {0.};

		if (b_state){
			std::ostringstream ost;
			ost << id  << "_" << (_t_cur-_dt);
			ost << "_" << _sys.P();
			string fn("mesh_" + ost.str());

			std::string model_path = _model_name;
			boost::filesystem::path path(model_path);

			// MdK 27/01/2017. grid file is now created in the cwd of the program and
			// not in the directory where the mesh resides.
			const std::string dirname = path.filename().string() + "_mesh";

			if (! boost::filesystem::exists(dirname) ){
				boost::filesystem::create_directory(dirname);
			}
			std::ofstream ofst(dirname + "/" + fn);
			std::vector<std::ostream*> vec_str{&ofst};
			_sys.Dump(vec_str);
		}
		return MPILib::AlgorithmGrid(array_state,array_interpretation);
	}

	template <class WeightValue, class Solver>
	void MeshAlgorithm<WeightValue,Solver>::reportDensity() const
	{
		std::ostringstream ost;
		ost << _node_id  << "_" << _t_cur;
		ost << "_" << _sys.P();
		string fn("mesh_" + ost.str());

		std::string model_path = _model_name;
		boost::filesystem::path path(model_path);

		// MdK 27/01/2017. grid file is now created in the cwd of the program and
		// not in the directory where the mesh resides.
		const std::string dirname = path.filename().string() + "_mesh";

		if (! boost::filesystem::exists(dirname) ){
			boost::filesystem::create_directory(dirname);
		}
		std::ofstream ofst(dirname + "/" + fn);
			std::vector<std::ostream*> vec_str{&ofst};
			_sys.Dump(vec_str);
	}

	template <class WeightValue, class Solver>
	void MeshAlgorithm<WeightValue,Solver>::evolveNodeState
	(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
		MPILib::Time time
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
				_sys.RemapReversal();
	    }

	    // master equation
	    _p_master->Apply(_n_steps*_dt,_vec_rates,_vec_map);

	    _sys.RedistributeProbability(_n_steps);

 	    _t_cur += _n_steps*_dt;
 	    _rate = (_sys.*_sysfunction)()[0];
 	    _n_evolve++;
	}

	template <class WeightValue, class Solver>
	void MeshAlgorithm<WeightValue,Solver>::FillMap(const std::vector<WeightValue>& vec_weights)
	{
		// The order in which they weight matrices appear in the vec_weights vector is not related
		// to the order in which the matrices are stored in the master equation solver.
		// the former is determined by the order in which the connections to the node that this
		// algorithm belongs to. The latter is determined by the order in which the user presents the TransitionMatrix
		// objects to the Master object. For the user there is no obvious way to keep this in synchrony, so
		// we allow any order, and use the synaptic efficacy of a connection to find the corresponding matrix.

 		// this function will only be called once;
		_vec_map = std::vector<MPILib::Index>(vec_weights.size(),std::numeric_limits<MPILib::Index>::max());

 		for(MPILib::Index i_weight = 0; i_weight < _vec_map.size(); i_weight++){
			for (MPILib::Index i_mat = 0; i_mat < _p_master->NrMatrices(0); i_mat++){
				if ( fabs( _p_master->Efficacy(0,i_mat) - vec_weights[i_weight]._efficacy) < _tolerance ){
					if (_vec_map[i_weight] == std::numeric_limits<MPILib::Index>::max())
						_vec_map[i_weight] = i_mat;
					else {
						throw TwoDLib::TwoDLibException("There are two matrices associated with this weight.");
					}
				}
			}
			if (_vec_map[i_weight] == std::numeric_limits<MPILib::Index>::max()){
				throw TwoDLib::TwoDLibException("There are no matrices associated with this weight.");
			}
		}

 		_vec_rates = std::vector< std::vector<MPILib::Efficacy> >(0); // MeshAlgorithm really only uses the first array, i.e. the rates it receives in prepareEvole
 		_vec_rates.push_back( std::vector<MPILib::Efficacy>(vec_weights.size(),0.) );

	}

	template <class WeightValue,class Solver>
	void MeshAlgorithm<WeightValue,Solver>::prepareEvolve
	(
		const std::vector<MPILib::Rate>& nodeVector,
		const std::vector<WeightValue>& weightVector,
		const std::vector<MPILib::NodeType>& typeVector
	)
	{
		if (_vec_map.size() == 0)
			FillMap(weightVector);
		// take into account the number of connections

		assert(nodeVector.size() == weightVector.size());
		for (MPILib::Index i = 0; i < nodeVector.size(); i++){
			_vec_rates[0][i] = nodeVector[i]*weightVector[i]._number_of_connections;
		}
	}
}

#endif // include guard
