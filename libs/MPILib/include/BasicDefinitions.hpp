/*
 * BasicDefinitions.hpp
 *
 *  Created on: 29.05.2012
 *      Author: david
 */

#ifndef MPILIB_BASICDEFINITIONS_HPP_
#define MPILIB_BASICDEFINITIONS_HPP_

#include <MPILib/include/TypeDefinitions.hpp>


namespace MPILib {

#ifdef _INVESTIGATE_ALGORITHM
#define VALUE_ARG vec_value,
#define VALUE_MEMBER_ARG _vec_value,
#define VALUE_REF vector<ReportValue>&,
#define VALUE_REF_INIT vector<ReportValue>& vec_value,
#define VALUE_MEMBER vector<ReportValue> _vec_value;
#define VALUE_MEMBER_REF vector<ReportValue>& _vec_value;
#define VALUE_MEMBER_INIT _vec_value(vec_value),
#define VALUE_RETURN		virtual vector<ReportValue> GetValues() const { return _vec_value; }
#else
#define VALUE_ARG
#define VALUE_MEMBER_ARG
#define VALUE_REF
#define VALUE_REF_INIT
#define VALUE_MEMBER
#define VALUE_MEMBER_REF
#define VALUE_MEMBER_INIT
#define VALUE_RETURN
#endif

//from basicdefinitions
const int HAVE_ROOT = 1;


// RootRortHandler will start with initial display of the first TIME_ROOT_INITIAL_DISPLAY_SECONDS , by default
const Time TIME_ROOT_INITIAL_DISPLAY = 0.05;

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

const int MAXIMUM_NUMBER_CIRCULANT_BINS = 100000;
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



const int CANVAS_X_DIMENSION = 800;
const int CANVAS_Y_DIMENSION = 800;

} //end namespace

#endif /* MPILIB_BASICDEFINITIONS_HPP_ */
