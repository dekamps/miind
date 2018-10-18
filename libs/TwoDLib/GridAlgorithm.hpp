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
#include "MeshAlgorithm.hpp"

namespace TwoDLib {

/**
 * \brief Mesh or 2D algorithm class.
 *
 * This class simulates the evolution of a neural population density function on a 2D grid.
 */

	template <class WeightValue, class Solver=TwoDLib::MasterOMP>
	class GridAlgorithm : public MeshAlgorithm<WeightValue,Solver>  {

	public:
    GridAlgorithm
		(
			const std::string&, 		    	 //!< model file name
			const std::vector<std::string>&,     //!< collection of transition matrix files
			MPILib::Time,                        //!< default time step for Master equation
			MPILib::Time tau_refractive = 0,     //!< absolute refractive period
			const string& ratemethod = ""        //!< firing rate computation; by default the mass flux across threshold
		);

		GridAlgorithm(const GridAlgorithm&);


		/**
		 * Cloning operation, to provide each DynamicNode with its own
		 * Algorithm instance. Clients use the naked pointer at their own risk.
		 */
		virtual GridAlgorithm* clone() const;
  };
}

#endif
