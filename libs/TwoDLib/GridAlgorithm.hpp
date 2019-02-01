#ifndef _CODE_LIBS_TWODLIB_GRIDALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDALGORITHM_INCLUDE_GUARD

#include <string>
#include <vector>
#include <MPILib/include/AlgorithmInterface.hpp>
#include <MPILib/include/DelayedConnectionQueue.hpp>
#include <MPILib/include/CustomConnectionParameters.hpp>
#include "MasterOdeint.hpp"
#include "MasterOMP.hpp"
#include "Ode2DSystemGroup.hpp"
#include "pugixml.hpp"
#include "display.hpp"
#include "MasterGrid.hpp"

namespace TwoDLib {

/**
 * \brief Mesh or 2D algorithm class.
 *
 * This class simulates the evolution of a neural population density function on a 2D grid.
 */

	class GridAlgorithm : public DensityAlgorithmInterface<CustomConnectionParameters>{

	public:
    GridAlgorithm
		(
			const std::string&, 		    	 //!< model file name
			const std::string&,     //!< Transform matrix
			MPILib::Time,                        //!< default time step for Master equation
			double,
			double,
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

		virtual void assignNodeId( MPILib::NodeId );

	  virtual MPILib::AlgorithmGrid getGrid(MPILib::NodeId, bool b_state = true) const;

		virtual void reportDensity() const;

		virtual void setupMasterSolver(double cell_width);

		virtual void prepareEvolve(const std::vector<MPILib::Rate>& nodeVector,
				const std::vector<CustomConnectionParameters>& weightVector,
				const std::vector<MPILib::NodeType>& typeVector);

		using MPILib::AlgorithmInterface<CustomConnectionParameters>::evolveNodeState;
		virtual void evolveNodeState(const std::vector<MPILib::Rate>& nodeVector,
				const std::vector<CustomConnectionParameters>& weightVector, MPILib::Time time);

		virtual void applyMasterSolver(std::vector<MPILib::Rate> rates);

		void InitializeDensity(MPILib::Index i, MPILib::Index j){_sys.Initialize(0,i,j);}

		const Ode2DSystemGroup& Sys() const {return _sys; }

		std::vector<std::vector<TwoDLib::Redistribution>> ReversalMap() const { return _vec_vec_rev; }

		std::vector<std::vector<TwoDLib::Redistribution>> ResetMap() const { return _vec_vec_res; }

	protected:

		const std::string _model_name;
		const std::string _rate_method;

		MPILib::Rate _rate;
		MPILib::Time _t_cur;

		pugi::xml_document _doc;
		pugi::xml_node _root;

		std::vector<TwoDLib::Mesh> _vec_mesh;

		std::vector<std::vector<TwoDLib::Redistribution>> _vec_vec_rev;
		std::vector<std::vector<TwoDLib::Redistribution>> _vec_vec_res;

		std::vector<MPILib::Time>    _vec_tau_refractive;

		MPILib::Time _dt;

		TwoDLib::Ode2DSystemGroup _sys;

		MPILib::NodeId _node_id;

		std::unique_ptr<MasterGrid>   _p_master;
		MPILib::Number _n_evolve;
		MPILib::Number _n_steps;

		std::vector<std::vector<MPILib::DelayedConnectionQueue>> _vec_vec_delay_queues;

		TransitionMatrix 							_transformMatrix;
		CSRMatrix*										_csr_transform;
		vector<double>								_mass_swap;
		vector<double>								_efficacy_map;

		std::string _transform_matrix;

		double _start_v;
		double _start_w;

		const vector<double>& (TwoDLib::Ode2DSystemGroup::*_sysfunction) () const;

	protected:

		virtual void FillMap(const std::vector<CustomConnectionParameters>& weightVector);
		std::vector<Mesh> CreateMeshObject();
		pugi::xml_node CreateRootNode(const std::string&);
		std::vector<TwoDLib::Redistribution> Mapping(const std::string&);

  };
}

#endif
