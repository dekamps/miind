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
		const std::vector<std::string>& mat_names,
		MPILib::Time h,
		MPILib::Time tau_refractive,
		const std::string&  rate_method
	):MeshAlgorithm<WeightValue,Solver>(model_name, mat_names,h,tau_refractive,rate_method)
	{}

	template <class WeightValue, class Solver>
	GridAlgorithm<WeightValue,Solver>::GridAlgorithm(const GridAlgorithm<WeightValue,Solver>& rhs):
  MeshAlgorithm<WeightValue,Solver>(rhs)
	{}

  template <class WeightValue,class Solver>
	GridAlgorithm<WeightValue,Solver>* GridAlgorithm<WeightValue,Solver>::clone() const
	{
	  return new GridAlgorithm<WeightValue,Solver>(*this);
	}
}

#endif
