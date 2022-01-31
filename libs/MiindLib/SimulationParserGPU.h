#ifndef _INCLUDE_GUARD_SIMULATION_PARSER_GPU
#define _INCLUDE_GUARD_SIMULATION_PARSER_GPU

#include <string>
#include <MPILib/include/MiindTvbModelAbstract.hpp>
#include <MPILib/include/RateAlgorithm.hpp>
#include <TwoDLib/GridAlgorithm.hpp>
#include <MPILib/include/CustomConnectionParameters.hpp>
#include <CudaTwoDLib/CudaTwoDLib.hpp>
#include <MiindLib/VectorizedNetwork.hpp>
#include <TwoDLib/SimulationParserCPU.h>

typedef CudaTwoDLib::fptype fptype;

template <class WeightType>
class SimulationParserGPU : public SimulationParserCPU<WeightType> {
public:
	SimulationParserGPU(int num_nodes, const std::string xml_filename, std::map<std::string,std::string> variables);
	SimulationParserGPU(const std::string xml_filename, std::map<std::string, std::string> variables);
	SimulationParserGPU(int num_nodes, const std::string xml_filename);
	SimulationParserGPU(const std::string xml_filename);

	void endSimulation();

	void addGridConnection(pugi::xml_node& xml_conn);

	void addMeshConnection(pugi::xml_node& xml_conn);

	bool addGridAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name);

	bool addMeshAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name);

	bool addRateFunctorNode(pugi::xml_document& doc, std::string alg_name);

	void addIncomingGridConnection(pugi::xml_node& xml_conn);

	void addIncomingMeshConnection(pugi::xml_node& xml_conn);

	void parseXmlFile();

	void startSimulation(TwoDLib::Display *display);

	void init();

	std::vector<double> evolveSingleStep(std::vector<double> activity);

	bool simulationComplete();

	int interpretValueAsInt(std::string value);

private:

	MiindLib::VectorizedNetwork vec_network;
	std::map<std::string, std::string> _node_algorithm_types;

	std::vector<TwoDLib::Mesh> _meshes;
	std::vector<std::vector<TwoDLib::Redistribution>> _reversal_mappings;
	std::vector<std::vector<TwoDLib::Redistribution>> _reset_mappings;
	std::vector<TwoDLib::TransitionMatrix> _transition_mats;
	std::map<std::string, std::map<double, TwoDLib::TransitionMatrix>> _mesh_transition_matrics;
	std::map<std::string, std::string> _node_algorithm_mapping;
	std::vector<rate_functor> _rate_functors;

	unsigned int _external_node_count;

};

#endif