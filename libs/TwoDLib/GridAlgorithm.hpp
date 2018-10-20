#ifndef _CODE_LIBS_TWODLIB_GRIDALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDALGORITHM_INCLUDE_GUARD

#include <string>
#include <vector>
#include <MPILib/include/AlgorithmInterface.hpp>
#include "MasterOdeint.hpp"
#include "MasterOMP.hpp"
#include "Ode2DSystem.hpp"
#include "pugixml.hpp"
#include "display.hpp"
#include "MasterGrid.hpp"

namespace TwoDLib {

/**
 * \brief Mesh or 2D algorithm class.
 *
 * This class simulates the evolution of a neural population density function on a 2D grid.
 */

	template <class WeightValue>
	class GridAlgorithm : public MPILib::AlgorithmInterface<WeightValue>{

	public:
    GridAlgorithm
		(
			const std::string&, 		    	 //!< model file name
			const std::string&,     //!< Transform matrix
			MPILib::Time,                        //!< default time step for Master equation
			MPILib::Index,
			MPILib::Index,
			MPILib::Time tau_refractive = 0,     //!< absolute refractive period
			const string& ratemethod = ""       //!< firing rate computation; by default the mass flux across threshold
		);

		GridAlgorithm(const GridAlgorithm&);


		/**
		 * Cloning operation, to provide each DynamicNode with its own
		 * Algorithm instance. Clients use the naked pointer at their own risk.
		 */
		virtual GridAlgorithm* clone() const;

		virtual void configure(const MPILib::SimulationRunParameter& simParam);

		virtual MPILib::Time getCurrentTime() const {return _t_cur;}

	  virtual MPILib::Rate getCurrentRate() const {return _rate;}

	  virtual MPILib::AlgorithmGrid getGrid(MPILib::NodeId, bool b_state = true) const;

		virtual void prepareEvolve(const std::vector<MPILib::Rate>& nodeVector,
				const std::vector<WeightValue>& weightVector,
				const std::vector<MPILib::NodeType>& typeVector);

		virtual void evolveNodeState(const std::vector<MPILib::Rate>& nodeVector,
				const std::vector<WeightValue>& weightVector, MPILib::Time time,
				const std::vector<MPILib::NodeType>& typeVector);

		void InitializeDensity(MPILib::Index i, MPILib::Index j){_sys.Initialize(i,j);}

		const Ode2DSystem& Sys() const {return _sys; }

		std::vector<TwoDLib::Redistribution> ReversalMap() const { return _vec_rev; }

		std::vector<TwoDLib::Redistribution> ResetMap() const { return _vec_res; }


	private:

		const std::string _model_name;
		const std::string _rate_method;

		MPILib::Rate _rate;
		MPILib::Time _t_cur;

		pugi::xml_document _doc;
		pugi::xml_node _root;

		TwoDLib::Mesh _mesh;

		std::vector<TwoDLib::Redistribution> _vec_rev;
		std::vector<TwoDLib::Redistribution> _vec_res;

		MPILib::Time _dt;

		TwoDLib::Ode2DSystem _sys;

		std::unique_ptr<MasterGrid>   _p_master;
		MPILib::Number _n_evolve;
		MPILib::Number _n_steps;

		std::vector<MPILib::Rate> _vec_rates;

		TransitionMatrix 							_transformMatrix;
		CSRMatrix*										_csr_transform;
		vector<double>								_mass_swap;
		vector<double>								_efficacy_map;

		std::string _transform_matrix;

		MPILib::Index _start_strip;
		MPILib::Index _start_cell;

		double (TwoDLib::Ode2DSystem::*_sysfunction) () const;

	private:

		void FillMap(const std::vector<WeightValue>& weightVector);
		Mesh CreateMeshObject();
		pugi::xml_node CreateRootNode(const std::string&);
		std::vector<TwoDLib::Redistribution> Mapping(const std::string&);

  };
}

#endif
