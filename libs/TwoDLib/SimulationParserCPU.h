#ifndef _INCLUDE_GUARD_SIMULATION_PARSER_CPU
#define _INCLUDE_GUARD_SIMULATION_PARSER_CPU

#include <string>
#include <MPILib/include/MiindTvbModelAbstract.hpp>
#include <MPILib/include/RateAlgorithm.hpp>
#include <TwoDLib/GridAlgorithm.hpp>
#include <TwoDLib/MeshAlgorithmCustom.hpp>
#include <MPILib/include/CustomConnectionParameters.hpp>


template <class WeightType>
class SimulationParserCPU : public MPILib::MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution> {
public:
	SimulationParserCPU(int num_nodes, const std::string xml_filename);
	SimulationParserCPU(const std::string xml_filename);
	void endSimulation();
	void addConnection(pugi::xml_node& xml_conn);
	void addIncomingConnection(pugi::xml_node& xml_conn);
	void parseXmlFile();
	void startSimulation();
	void init();
	double getCurrentSimTime();
	std::vector<double> evolveSingleStep(std::vector<double> activity);
	bool simulationComplete();

	std::vector<std::string>& getOrderedOutputNodes() { return _ordered_output_nodes; }
	unsigned int getIndexOfOutputNode(std::string node) { return find(_ordered_output_nodes.begin(), _ordered_output_nodes.end(), node) - _ordered_output_nodes.begin(); }

protected:

	std::string interpretValueAsString(std::string value);
	double interpretValueAsDouble(std::string value);
	int interpretValueAsInt(std::string value);

	std::map<std::string, std::string> _variables;

	void parseXMLAlgorithms(pugi::xml_document& doc,
		std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<WeightType>>>& _algorithms,
		std::map<std::string, MPILib::NodeId>& _node_ids);

	bool checkWeightType(pugi::xml_document& doc);

	std::string _xml_filename;

	std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<WeightType>>> _algorithms;
	std::map<std::string, MPILib::NodeId> _node_ids;

	unsigned long _count;
	std::vector<MPILib::NodeId> _display_nodes;
	std::vector<MPILib::NodeId> _rate_nodes;
	std::vector<MPILib::Time> _rate_node_intervals;
	std::vector<MPILib::NodeId> _density_nodes;
	std::vector<MPILib::Time> _density_node_start_times;
	std::vector<MPILib::Time> _density_node_end_times;
	std::vector<MPILib::Time> _density_node_intervals;

	std::vector<std::string> _ordered_output_nodes;
};

#endif