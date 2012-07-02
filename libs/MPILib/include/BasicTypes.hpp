/*
 * BasicTypes.hpp
 *
 *  Created on: 29.05.2012
 *      Author: david
 */

#ifndef MPILIB_BASICTYPES_HPP_
#define MPILIB_BASICTYPES_HPP_

#include <string>
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

typedef double Rate;
typedef Rate ActivityType;
typedef double Time;
typedef Time TimeStep;
typedef double Density;
typedef double Potential;
typedef double Efficacy;

typedef unsigned int Number;
typedef unsigned int Index;

// RootRortHandler will start with initial display of the first TIME_ROOT_INITIAL_DISPLAY_SECONDS , by default
const Time TIME_ROOT_INITIAL_DISPLAY = 0.05;

const double EPSILON_INTEGRALRATE = 1e-4;


//! Rate Algorithm nodes have a single state
const int RATE_STATE_DIMENSION = 1;

//! Wilson Cowan nodes have single double as state
const int WILSON_COWAN_STATE_DIMENSION = 1;

//my own
typedef int NodeId;


//! The parameter vector for Wilson Cowan integration has four elements
const int WILSON_COWAN_PARAMETER_DIMENSION = 4;

const double WC_ABSOLUTE_PRECISION = 1e-5;
const double WC_RELATIVE_PRECISION = 0;

const int N_FRACT_PERCENTAGE_SMALL = 100;
const int N_PERCENTAGE_SMALL       = 5;
const int N_FRACT_PERCENTAGE_BIG   = 10;

const int KEY_PRECISION = 8;

const std::string STR_ASCIIHANDLER_EXCEPTION
				(
					"Could not open ascii handler file stream:"
				);

const std::string STR_HANDLER_STALE
				(
					"This handler already has written reports"
				);

const std::string STR_DYNAMICLIB_EXCEPTION
				(
					"Some DynamicLib exception occurred"
				);

const std::string STR_STATE_CONFIGURATION_EXCEPTION
				(
					"There is a mismatch between the dimension of the State and the EvolutionAlgorithm"
				);

const std::string STR_ROOTFILE_EXCEPTION
				(
					"Couldn't open root file"
				);

const std::string STR_AE_TAG
				(
					"<AbstractAlgorithm>"
				);

const std::string STR_AE_EXCEPTION
				(
					"Can't serialize an AbstractAlgorithm"
				);

const std::string STR_NETWORKSTATE_TAG
				(
					"<NetworkState>"
				);

const std::string STR_RA_TAG
				(
					"<RateAlgorithm>"
				);

const std::string STR_NETWORKSTATE_ENDTAG
				(
					"</NetworkState>"
				);

const std::string STRING_WC_TAG
				(
					"<WilsonCowanAlgorithm>"
				);
const std::string STR_DYNAMICNETWORKIMPLEMENTATION_TAG
				(
					"<DynamicNetworkImplementation>"
				);
const std::string STR_OPENNETWORKFILE_FAILED
				(
					"Could not open test file. Does test directory exist ?"
				);

const std::string STR_DYNAMIC_NETWORK_TAG
				(
					"<DynamicNetwork>"
				);
const std::string STR_NETWORK_CREATION_FAILED
				(
					"Creation of test dynamic network failed"
				);
const std::string STR_EXCEPTION_CAUSE_UNKNOWN
				(
					"Unknow exception thrown in Evolve"
				);
const std::string STR_NUMBER_ITERATIONS_EXCEEDED
				(
					"The predetermined number of iterations is exceeded in Evolve()"
				);
const std::string STR_INCOMPATIBLE_TIMING_ERROR
				(
					"Node Evolve algorithm didn't reach specified end time"
				);
const std::string STR_TIME
				(
					"<Time>"
				);
const std::string STR_NODEID
				(
					"<NodeId>"
				);
const std::string STR_REPORT
				(
					"<Report>"
				);

const std::string STR_GRID_TAG
				(
					"<AlgorithmGrid>"
				);
const std::string STR_GRID_PARSE_ERROR
				(
					"Error parsing AlgorithmGrid"
				);
const std::string STR_STATE_PARSE_ERROR
				(
					"Error parsing NodeState"
				);
const std::string STR_NODESTATE_TAG
				(
					"<NodeState>"
				);
const std::string STR_DYN_TAG
				(
					"<DynamicNode>"
				);
const std::string STR_WCP_TAG
				(
					"<WilsonCowanParameter>"
				);
const std::string STRING_NODEVALUE
				(
					"<NodeValue>"
				);

const std::string CANVAS_NAME
				(
					"Canvas"
				);

const std::string CANVAS_TITLE
				(
					"Dynamic population overview"
				);

const std::string STR_NETWORK_IMPLEMENTATION_REALLOC
				(
					"Network implementation will realloc. This invalidates the internal representation"
				);

const std::string STR_ROOT_FILE_OPENED_FAILED
				(
					"Could not open ROOT file"
				);

const std::string SIMRUNPAR_UNEXPECTED("Unexpected tag for SimulationRunParameter");

const std::string OFFSET_ERROR("Cast to DynamicNode failed");

const int CANVAS_X_DIMENSION = 800;
const int CANVAS_Y_DIMENSION = 800;

}//end namespace

#endif /* MPILIB_BASICTYPES_HPP_ */
