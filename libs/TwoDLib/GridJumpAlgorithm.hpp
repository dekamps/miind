#ifndef _CODE_LIBS_TWODLIB_GRIDJUMPALGORITHM_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDJUMPALGORITHM_INCLUDE_GUARD

#include <MPILib/include/CustomConnectionParameters.hpp>
#include "MasterGridJump.hpp"

namespace TwoDLib {

	class GridJumpAlgorithm : public GridAlgorithm{
	public:
		GridJumpAlgorithm
		(
			const std::string&, 		    	 //!< model file name
			const std::string&,     //!< Transform matrix
			MPILib::Time,                        //!< default time step for Master equation
			double,
			double,
			MPILib::Time tau_refractive = 0,     //!< absolute refractive period
			const string& ratemethod = ""       //!< firing rate computation; by default the mass flux across threshold
		);

		GridJumpAlgorithm(const GridJumpAlgorithm&);

		virtual GridJumpAlgorithm* clone() const;

		virtual void prepareEvolve(const std::vector<MPILib::Rate>& nodeVector,
				const std::vector<CustomConnectionParameters>& weightVector,
				const std::vector<MPILib::NodeType>& typeVector);

	protected:
		virtual void setupMasterSolver(double cell_width);
		virtual void applyMasterSolver(std::vector<MPILib::Rate> rates);
		virtual void FillMap(const std::vector<CustomConnectionParameters>& weightVector);

		std::vector<double> _connection_stat_v;
		std::unique_ptr<MasterGridJump>   _p_master_jump;
	};

}

#endif
