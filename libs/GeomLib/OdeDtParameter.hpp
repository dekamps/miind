// Copyright (c) 2005 - 2014 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification,are permitted provided that the following conditions are
// met
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of
//      conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived
//      from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
// FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
// THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//
#ifndef  _CODE_LIBS_GEOMLIB_ODEDTPARAMETER
#define  _CODE_LIBS_GEOMLIB_ODEDTPARAMETER
#include <string>
#include "CurrentCompensationParameter.hpp"
#include "NeuronParameter.hpp"
#include "InitialDensityParameter.hpp"

using MPILib::Potential;
using MPILib::Number;
using std::string;

namespace GeomLib
{

  //! Contains the parameters necessary to configure a concrete OdeSystem instance. See AbstractOdeSystem and derived classes.
  //! An AbstractOdeSystem is geometric grid: a grid defined by a system of ordinary differential equations.
  //! The grid needs dimensions: a minimum potential specified by _V_min, a maximum defined in the NeuronParameter
  //! as the threshold, and an initial density at the start of the simulation, as well as a time step.
  //! The time step implicitly defines the number of bins: if you want a finer grid, take a smaller time step.
  //! The time step directly determines the the number of bins between reversal potential and threshold. The
  //! number of negative bins follows is calculated similarly, but extrapolated to V_min rather than -V_threshold.
  //! For GeomAlgorithm, it is essential that all grids use the same time step, therefore OdeParameter, where
  //! you could specify the number of bins, and where the time step would be calculated will be deprecated.



    struct OdeDtParameter {

      MPILib::Time              _dt;       //!< The time step, implicitly determining the number of bins
      InitialDensityParameter   _par_dens; //!< Specifies the initial density profile
      Potential                 _V_min;    //!< The minimum of potential range (the maximum is given in the neuron parameter)
      NeuronParameter      	    _par_pop;  //!< The neuron parameter

      OdeDtParameter
      (
       MPILib::Time,                  //!< Time step of the geometric grid
       Potential,                     //!< V_min, the minimum value of the Potential range
       const NeuronParameter&,    	  //!< Neuron parameter
       const InitialDensityParameter& //!< Initial density profile
      );

    };

} // namespace GeomLib
#endif // end of ODEDTPARAMETER

