/*
 * BasicTypes.hpp
 *
 *  Created on: 29.05.2012
 *      Author: david
 */

#ifndef MPILIB_BASICTYPES_HPP_
#define MPILIB_BASICTYPES_HPP_

#include <MPILib/include/NodeState.hpp>


namespace MPILib {

//from basicdefinitions
const int HAVE_ROOT = 1;

typedef double Rate;
typedef double Time;
typedef Time TimeStep;
typedef double Density;
typedef double Potential;
typedef double Efficacy;

// RootRortHandler will start with initial display of the first TIME_ROOT_INITIAL_DISPLAY_SECONDS , by default
const Time TIME_ROOT_INITIAL_DISPLAY = 0.05;
//from localdefinitions
typedef double Rate;
typedef double Time;
typedef Time TimeStep;

//! Rate Algorithm nodes have a single state
const int RATE_STATE_DIMENSION = 1;

//! Wilson Cowan nodes have single double as state
const int WILSON_COWAN_STATE_DIMENSION = 1;

//! The parameter vector for Wilson Cowan integration has four elements
const int WILSON_COWAN_PARAMETER_DIMENSION = 4;

const double WC_ABSOLUTE_PRECISION = 1e-5;
const double WC_RELATIVE_PRECISION = 0;

const int N_FRACT_PERCENTAGE_SMALL = 100;
const int N_PERCENTAGE_SMALL = 5;
const int N_FRACT_PERCENTAGE_BIG = 10;

const int KEY_PRECISION = 8;

//my own
typedef int NodeType;
typedef int NodeId;
typedef int SimulationRunParameter;

const string STR_NUMBER_ITERATIONS_EXCEEDED
				(
					"The predetermined number of iterations is exceeded in Evolve()"
				);

}//end namespace

#endif /* MPILIB_BASICTYPES_HPP_ */
