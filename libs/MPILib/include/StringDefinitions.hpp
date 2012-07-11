/*
 * StringDefinitions.hpp
 *
 *  Created on: 11.07.2012
 *      Author: david
 */

#ifndef MPILIB_STRINGDEFINITIONS_HPP_
#define MPILIB_STRINGDEFINITIONS_HPP_

#include <string>

namespace MPILib{

const std::string STR_BINS_MUST_BE_ADDED(
		"You have specified a finite time constant, but no bins to be added. This is inconsistent");

const std::string STR_MEMBRANE_ZERO(
		"You have taken the membrane time constant equal to zero. Evolution can not proceed");

const std::string STR_ASCIIHANDLER_EXCEPTION(
		"Could not open ascii handler file stream:");

const std::string STR_HANDLER_STALE("This handler already has written reports");

const std::string STR_DYNAMICLIB_EXCEPTION(
		"Some DynamicLib exception occurred");

const std::string STR_STATE_CONFIGURATION_EXCEPTION(
		"There is a mismatch between the dimension of the State and the EvolutionAlgorithm");

const std::string STR_ROOTFILE_EXCEPTION("Couldn't open root file");

const std::string STR_AE_TAG("<AbstractAlgorithm>");

const std::string STR_AE_EXCEPTION("Can't serialize an AbstractAlgorithm");

const std::string STR_NETWORKSTATE_TAG("<NetworkState>");

const std::string STR_RA_TAG("<RateAlgorithm>");

const std::string STR_NETWORKSTATE_ENDTAG("</NetworkState>");

const std::string STRING_WC_TAG("<WilsonCowanAlgorithm>");
const std::string STR_DYNAMICNETWORKIMPLEMENTATION_TAG(
		"<DynamicNetworkImplementation>");
const std::string STR_OPENNETWORKFILE_FAILED(
		"Could not open test file. Does test directory exist ?");

const std::string STR_DYNAMIC_NETWORK_TAG("<DynamicNetwork>");
const std::string STR_NETWORK_CREATION_FAILED(
		"Creation of test dynamic network failed");
const std::string STR_EXCEPTION_CAUSE_UNKNOWN(
		"Unknow exception thrown in Evolve");
const std::string STR_NUMBER_ITERATIONS_EXCEEDED(
		"The predetermined number of iterations is exceeded in Evolve()");
const std::string STR_INCOMPATIBLE_TIMING_ERROR(
		"Node Evolve algorithm didn't reach specified end time");
const std::string STR_TIME("<Time>");
const std::string STR_NODEID("<NodeId>");
const std::string STR_REPORT("<Report>");

const std::string STR_GRID_TAG("<AlgorithmGrid>");
const std::string STR_GRID_PARSE_ERROR("Error parsing AlgorithmGrid");
const std::string STR_STATE_PARSE_ERROR("Error parsing NodeState");
const std::string STR_NODESTATE_TAG("<NodeState>");
const std::string STR_DYN_TAG("<DynamicNode>");
const std::string STR_WCP_TAG("<WilsonCowanParameter>");
const std::string STRING_NODEVALUE("<NodeValue>");

const std::string CANVAS_NAME("Canvas");

const std::string CANVAS_TITLE("Dynamic population overview");

const std::string STR_NETWORK_IMPLEMENTATION_REALLOC(
		"Network implementation will realloc. This invalidates the internal representation");

const std::string STR_ROOT_FILE_OPENED_FAILED("Could not open ROOT file");

const std::string WORKSPACE_EXCESSION(
		"Interpolation workspace exceeded. Recompile if necessary");

const std::string SIMRUNPAR_UNEXPECTED(
		"Unexpected tag for SimulationRunParameter");

const std::string OFFSET_ERROR("Cast to DynamicNode failed");

}//end namespace MPILib

#endif /* MPILIB_STRINGDEFINITIONS_HPP_ */
