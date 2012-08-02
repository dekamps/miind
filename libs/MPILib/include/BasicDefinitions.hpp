// Copyright (c) 2005 - 2012 Marc de Kamps
//						2012 David-Matthias Sichau
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

#ifndef MPILIB_BASICDEFINITIONS_HPP_
#define MPILIB_BASICDEFINITIONS_HPP_

#include <MPILib/include/TypeDefinitions.hpp>


namespace MPILib {


//from basicdefinitions
const int HAVE_ROOT = 1;

const double EPSILON_INTEGRALRATE = 1e-4;

//! Rate Algorithm nodes have a single state
const int RATE_STATE_DIMENSION = 1;

//! Wilson Cowan nodes have single double as state
const int WILSON_COWAN_STATE_DIMENSION = 1;
//! if a double input must mimmick a diffusion process, then a small efficacy must be chosen
const double DIFFUSION_STEP = 0.03;
//! if a stepsize is calculated to larger than DIFFUSION_LIMIT\f$ (\theta -V_{rev})\f$, a single input cannot mimmck a diffusion process any more
const double DIFFUSION_LIMIT = 0.05;

//! default maximum term size for AbstractNonCirculantSolver
const double EPS_J_CIRC_MAX = 1e-10;

//TODO: restore
const double RELATIVE_LEAKAGE_PRECISION = 1e-4; //1e-10;

const double ALPHA_LIMIT = 1e-6;

const unsigned int MAXIMUM_NUMBER_CIRCULANT_BINS = 100000;
const int MAXIMUM_NUMBER_NON_CIRCULANT_BINS = 100000;

//! The parameter vector for Wilson Cowan integration has four elements
const int WILSON_COWAN_PARAMETER_DIMENSION = 4;

const double WC_ABSOLUTE_PRECISION = 1e-5;
const double WC_RELATIVE_PRECISION = 0;

const int N_FRACT_PERCENTAGE_SMALL = 100;
const int N_PERCENTAGE_SMALL = 5;
const int N_FRACT_PERCENTAGE_BIG = 10;

const Number CIRCULANT_POLY_DEGREE = 4;

const Index CIRCULANT_POLY_JMAX = 7;

const int MAXIMUM_NUMBER_GAMMAZ_VALUES = 500;

const Number NUMBER_INTEGRATION_WORKSPACE = 10000;

//! a large value, but not numeric_limits::max, since this value is used in divisions
//! this corresponds to a membrane time constant of twenty minuts
const Time TIME_MEMBRANE_INFINITE = 1000000;

const Number NONCIRC_LIMIT = 5;

//! Even if refraction is not considered, for some algorithms it is convenient to set it artificially to a non zero value
const Time TIME_REFRACT_MIN = 0.1e-3;

const int KEY_PRECISION = 8;

const int MAX_V_ARRAY = 100000; // should be more than sufficient


} //end namespace

#endif /* MPILIB_BASICDEFINITIONS_HPP_ */
