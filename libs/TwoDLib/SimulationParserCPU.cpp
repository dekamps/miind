#include "SimulationParserCPU.h"
#include <map>
#include <memory>
#include <TwoDLib/GridReport.hpp>
#include <MPILib/include/WilsonCowanAlgorithm.hpp>
#include <TwoDLib/GridSomaDendriteAlgorithmCode.hpp>
#include <TwoDLib/GridJumpAlgorithmCode.hpp>

template<>
SimulationParserCPU<MPILib::CustomConnectionParameters>::SimulationParserCPU(int num_nodes, const std::string xml_filename, std::map<std::string, std::string> vars) :
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>(num_nodes, 1.0), _count(0), _xml_filename(xml_filename), _variables(vars) {
}

template<>
SimulationParserCPU<MPILib::CustomConnectionParameters>::SimulationParserCPU(const std::string xml_filename, std::map<std::string, std::string> vars) :
	SimulationParserCPU(1, xml_filename, vars) {
}

template<>
SimulationParserCPU<MPILib::CustomConnectionParameters>::SimulationParserCPU(int num_nodes, const std::string xml_filename) :
	SimulationParserCPU(num_nodes, xml_filename,std::map<std::string,std::string>()) {
}

template<>
SimulationParserCPU<MPILib::CustomConnectionParameters>::SimulationParserCPU(const std::string xml_filename) :
	SimulationParserCPU(1,xml_filename) {
}

template<>
SimulationParserCPU<MPILib::DelayedConnection>::SimulationParserCPU(int num_nodes, const std::string xml_filename, std::map<std::string, std::string> vars) :
	MiindTvbModelAbstract<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution>(num_nodes, 1.0), _count(0), _xml_filename(xml_filename), _variables(vars) {
}

template<>
SimulationParserCPU<MPILib::DelayedConnection>::SimulationParserCPU(const std::string xml_filename, std::map<std::string, std::string> vars) :
	SimulationParserCPU(1,xml_filename, vars) {
}

template<>
SimulationParserCPU<MPILib::DelayedConnection>::SimulationParserCPU(int num_nodes, const std::string xml_filename) :
	SimulationParserCPU(num_nodes, xml_filename, std::map<std::string,std::string>()) {
}

template<>
SimulationParserCPU<MPILib::DelayedConnection>::SimulationParserCPU(const std::string xml_filename) :
	SimulationParserCPU(1, xml_filename) {
}

template<>
SimulationParserCPU<double>::SimulationParserCPU(int num_nodes, const std::string xml_filename, std::map<std::string, std::string> vars) :
	// For now we don't allow num_nodes : override to 1 node only.
	MiindTvbModelAbstract<double, MPILib::utilities::CircularDistribution>(num_nodes, 1.0), _count(0), _xml_filename(xml_filename), _variables(vars) {
}

template<>
SimulationParserCPU<double>::SimulationParserCPU(const std::string xml_filename, std::map<std::string, std::string> vars) :
	SimulationParserCPU(1, xml_filename, vars){
}

template<>
SimulationParserCPU<double>::SimulationParserCPU(int num_nodes, const std::string xml_filename) :
	SimulationParserCPU(num_nodes, xml_filename, std::map<std::string,std::string>()) {
}

template<>
SimulationParserCPU<double>::SimulationParserCPU(const std::string xml_filename) :
	SimulationParserCPU(1,xml_filename) {
}


template<class WeightType >
std::string SimulationParserCPU<WeightType>::interpretValueAsString(std::string value) {
	
	if (_variables.find(value) == _variables.end()) // If the string isn't in the map, then assume it's just a string.
		return value;

	// todo: Do a sanity check that the variable was defined with type=String.

	return _variables[value];
}

template<class WeightType >
double SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string value) {
	
	if (value == "")
		return 0.0;

	// todo: Do some checking to see if this is an actual double

	if (_variables.find(value) == _variables.end()) // If the string isn't in the map, assume it's just a value
		return std::stod(value);

	if (_variables[value] == "")
		std::cout << "Warning: The value of variable " << value << " in xml file is empty and cannot be converted to a number.\n";

	return std::stod(_variables[value]);
}

template<class WeightType>
void SimulationParserCPU<WeightType>::endSimulation() {
	MPILib::MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::endSimulation();
}

template<>
void SimulationParserCPU<MPILib::CustomConnectionParameters>::addConnection(pugi::xml_node& xml_conn) {
	MPILib::CustomConnectionParameters connection;

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(_current_node);
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(_current_node);

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("In") == std::string(ait->name())) || (std::string("Out") == std::string(ait->name())))
			continue;

		connection.setParam(std::string(ait->name()), interpretValueAsString(std::string(ait->value())));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
	_connections.push_back(connection);
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.makeFirstInputOfSecond(_node_ids[in], _node_ids[out], _connections.back());
}

template<>
void SimulationParserCPU<MPILib::DelayedConnection>::addConnection(pugi::xml_node& xml_conn) {
	

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(_current_node);
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(_current_node);

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);
	
	MPILib::DelayedConnection connection(interpretValueAsDouble(std::string(num_connections)), interpretValueAsDouble(std::string(efficacy)), interpretValueAsDouble(std::string(delay)));
	_connections.push_back(connection);
	MiindTvbModelAbstract<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution>::network.makeFirstInputOfSecond(_node_ids[in], _node_ids[out], _connections.back());
}

template<>
void SimulationParserCPU<double>::addConnection(pugi::xml_node& xml_conn) {

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(_current_node);
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(_current_node);

	double value = interpretValueAsDouble(xml_conn.text().as_string());
	_connections.push_back(value);
	MiindTvbModelAbstract<double, MPILib::utilities::CircularDistribution>::network.makeFirstInputOfSecond(_node_ids[in], _node_ids[out], _connections.back());
}

template<>
void SimulationParserCPU<MPILib::CustomConnectionParameters>::addIncomingConnection(pugi::xml_node& xml_conn) {
	MPILib::CustomConnectionParameters connection;

	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(_current_node);

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection.setParam(std::string(ait->name()), interpretValueAsString(std::string(ait->value())));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
	_connections.push_back(connection);
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.setNodeExternalPrecursor(_node_ids[node], _connections.back());
}

template<>
void SimulationParserCPU<MPILib::DelayedConnection>::addIncomingConnection(pugi::xml_node& xml_conn) {
	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(_current_node);


	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);
	
	MPILib::DelayedConnection connection(interpretValueAsDouble(std::string(num_connections)), interpretValueAsDouble(std::string(efficacy)), interpretValueAsDouble(std::string(delay)));
	_connections.push_back(connection);
	MiindTvbModelAbstract<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution>::network.setNodeExternalPrecursor(_node_ids[node], _connections.back());
}

template<>
void SimulationParserCPU<double>::addIncomingConnection(pugi::xml_node& xml_conn) {
	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(_current_node);

	double value = interpretValueAsDouble(xml_conn.text().as_string());
	_connections.push_back(value);
	MiindTvbModelAbstract<double, MPILib::utilities::CircularDistribution>::network.setNodeExternalPrecursor(_node_ids[node], _connections.back());
}

template<class WeightType>
double SimulationParserCPU<WeightType>::getCurrentSimTime() {
	return _count * MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_time_step;
}

template<>
void SimulationParserCPU< MPILib::CustomConnectionParameters>::parseXMLAlgorithms(pugi::xml_document& doc,
	std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>>& _algorithms,
	std::map<std::string, MPILib::NodeId>& _node_ids) {

	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		//Check all possible Algorithm types
		if (std::string("GridAlgorithm") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found GridAlgorithm " << algorithm_name << ".\n";

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(algorithm.attribute("tau_refractive").as_string());
			std::string transform_filename = interpretValueAsString(std::string(algorithm.attribute("transformfile").value()));
			double start_v = interpretValueAsDouble(algorithm.attribute("start_v").as_string());
			double start_w = interpretValueAsDouble(algorithm.attribute("start_w").as_string());
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));
			std::string activity_mode = interpretValueAsString(std::string(algorithm.attribute("ratemethod").value()));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new TwoDLib::GridAlgorithm(model_filename, transform_filename, time_step, start_v, start_w, tau_refractive, activity_mode));
		}

		if (std::string("GridSomaDendriteAlgorithm") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found GridSomaDendriteAlgorithm " << algorithm_name << ".\n";

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(algorithm.attribute("tau_refractive").as_string());
			std::string transform_filename = interpretValueAsString(std::string(algorithm.attribute("transformfile").value()));
			double start_v = interpretValueAsDouble(algorithm.attribute("start_v").as_string());
			double start_w = interpretValueAsDouble(algorithm.attribute("start_w").as_string());
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));
			std::string activity_mode = interpretValueAsString(std::string(algorithm.attribute("ratemethod").value()));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new TwoDLib::GridSomaDendriteAlgorithm(model_filename, transform_filename, time_step, start_v, start_w, tau_refractive, activity_mode));
		}

		if (std::string("GridJumpAlgorithm") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found GridJumpAlgorithm " << algorithm_name << ".\n";

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(algorithm.attribute("tau_refractive").as_string());
			std::string transform_filename = interpretValueAsString(std::string(algorithm.attribute("transformfile").value()));
			double start_v = interpretValueAsDouble(algorithm.attribute("start_v").as_string());
			double start_w = interpretValueAsDouble(algorithm.attribute("start_w").as_string());
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));
			std::string activity_mode = interpretValueAsString(std::string(algorithm.attribute("ratemethod").value()));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new TwoDLib::GridJumpAlgorithm(model_filename, transform_filename, time_step, start_v, start_w, tau_refractive, activity_mode));
		}

		if (std::string("MeshAlgorithmCustom") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found MeshAlgorithmCustom " << algorithm_name << ".\n" << std::flush;

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(algorithm.attribute("tau_refractive").as_string());
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));
			std::string activity_mode = interpretValueAsString(std::string(algorithm.attribute("ratemethod").value()));

			std::vector<std::string> matrix_files;
			for (pugi::xml_node matrix_file = algorithm.child("MatrixFile"); matrix_file; matrix_file = matrix_file.next_sibling("MatrixFile")) {
				matrix_files.push_back(interpretValueAsString(std::string(matrix_file.child_value())));
			}

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new TwoDLib::MeshAlgorithmCustom<TwoDLib::MasterOdeint>(model_filename, matrix_files, time_step, tau_refractive, activity_mode));
		}

		if (std::string("RateFunctor") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just use RateAlgorithm for RateFunctor for now.
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found RateFunctor (Using a RateAlgorithm) " << algorithm_name << ".\n";

			double rate = interpretValueAsDouble(std::string(algorithm.child_value("expression")));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new MPILib::RateAlgorithm<MPILib::CustomConnectionParameters>(rate));
		}

	}

}

template<>
void SimulationParserCPU< MPILib::DelayedConnection>::parseXMLAlgorithms(pugi::xml_document& doc,
	std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<MPILib::DelayedConnection>>>& _algorithms,
	std::map<std::string, MPILib::NodeId>& _node_ids) {

	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		//Check all possible Algorithm types
		if (std::string("MeshAlgorithm") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found MeshAlgorithm " << algorithm_name << ".\n" << std::flush;

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(algorithm.attribute("tau_refractive").as_string());
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));
			std::string activity_mode = interpretValueAsString(std::string(algorithm.attribute("ratemethod").value()));

			std::vector<std::string> matrix_files;
			for (pugi::xml_node matrix_file = algorithm.child("MatrixFile"); matrix_file; matrix_file = matrix_file.next_sibling("MatrixFile")) {
				matrix_files.push_back(interpretValueAsString(std::string(matrix_file.child_value())));
			}

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::DelayedConnection>>(new TwoDLib::MeshAlgorithm<MPILib::DelayedConnection,TwoDLib::MasterOdeint>(model_filename, matrix_files, time_step, tau_refractive, activity_mode));
		}

		if (std::string("OUAlgorithm") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {

			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found OUAlgorithm " << algorithm_name << ".\n";

			double t_membrane = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("t_membrane")));
			double t_refractive = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("t_refractive")));
			double V_threshold = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("V_threshold")));
			double V_reset = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("V_reset")));
			double V_reversal = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("V_reversal")));

			GeomLib::NeuronParameter params(V_threshold, V_reset, V_reversal, t_refractive, t_membrane);

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::DelayedConnection>>(new GeomLib::OUAlgorithm(params));
		}

		if (std::string("RateFunctor") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just use RateAlgorithm for RateFunctor for now.
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found RateFunctor (Using a RateAlgorithm) " << algorithm_name << ".\n";

			double rate = interpretValueAsDouble(std::string(algorithm.child_value("expression")));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::DelayedConnection>>(new MPILib::RateAlgorithm<MPILib::DelayedConnection>(rate));
		}
	}

}

template<>
void SimulationParserCPU<double>::parseXMLAlgorithms(pugi::xml_document& doc,
	std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<double>>>& _algorithms,
	std::map<std::string, MPILib::NodeId>& _node_ids) {

	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		//Check all possible Algorithm types

		if (std::string("WilsonCowanAlgorithm") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {

			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found WilsonCowanAlgorithm " << algorithm_name << ".\n";

			double t_membrane = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("t_membrane")));
			double f_noise = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("f_noise")));
			double f_max = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("f_max")));
			double f_bias = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("f_bias")));
			double I_ext = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("I_ext")));
			double smoothing = interpretValueAsDouble(std::string(algorithm.child("NeuronParameter").child_value("smoothing")));

			MPILib::WilsonCowanParameter params(t_membrane, f_max, f_noise, f_bias, I_ext, smoothing);
			wcparams.push_back(params);
			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<double>>(new MPILib::WilsonCowanAlgorithm(wcparams.back()));
		}

		if (std::string("RateFunctor") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just use RateAlgorithm for RateFunctor for now.
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			std::cout << "Found RateFunctor (Using a RateAlgorithm) " << algorithm_name << ".\n";

			double rate = interpretValueAsDouble(std::string(algorithm.child_value("expression")));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<double>>(new MPILib::RateAlgorithm<double>(rate));
		}

	}

}

template<>
bool SimulationParserCPU<MPILib::CustomConnectionParameters>::checkWeightType(pugi::xml_document& doc) {
	if (std::string("CustomConnectionParameters") != std::string(doc.child("Simulation").child_value("WeightType"))) {
		std::cout << "The weight type of the SimulationParser (" << "CustomConnectionParameters" << ") doesn't match the WeightType in the XML file (" << doc.child("Simulation").child_value("WeightType") << "). Exiting.\n";
		return false;
	}
	return true;
}

template<>
bool SimulationParserCPU<MPILib::DelayedConnection>::checkWeightType(pugi::xml_document& doc) {
	if (std::string("DelayedConnection") != std::string(doc.child("Simulation").child_value("WeightType"))) {
		std::cout << "The weight type of the SimulationParser (" << "DelayedConnection" << ") doesn't match the WeightType in the XML file (" << doc.child("Simulation").child_value("WeightType") << "). Exiting.\n";
		return false;
	}
	return true;
}

template<>
bool SimulationParserCPU<double>::checkWeightType(pugi::xml_document& doc) {
	if (std::string("double") != std::string(doc.child("Simulation").child_value("WeightType"))) {
		std::cout << "The weight type of the SimulationParser (" << "double" << ") doesn't match the WeightType in the XML file (" << doc.child("Simulation").child_value("WeightType") << "). Exiting.\n";
		return false;
	}
	return true;
}

template<class WeightType>
void SimulationParserCPU<WeightType>::parseXmlFile() {
	pugi::xml_document doc;
	if (!doc.load_file(_xml_filename.c_str())) {
		std::cout << "Failed to load XML simulation file.\n";
		return; //better to throw...
	}

	//check Weight Type matches this class
	if (!checkWeightType(doc))
		return;

	// Load Variables into map - map might already have been populated with external variables
	// So only add variables which haven't been set externally
	for (pugi::xml_node var = doc.child("Simulation").child("Variable"); var; var = var.next_sibling("Variable")) {
		if (!_variables.count(std::string(var.attribute("Name").value())))
			_variables[std::string(var.attribute("Name").value())] = std::string(var.text().as_string());
	}

	//Algorithms
	//Ignore "Group" algorithms - this is the non-cuda version of WinMiind (for now)

	_algorithms = std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<WeightType>>>();
	_node_ids = std::map<std::string, MPILib::NodeId>();

	parseXMLAlgorithms(doc, _algorithms, _node_ids);

	// For now, let's allow multiple nodes but only with the same variables and algorithms.

	for (unsigned int node_num = 0; node_num < MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_num_nodes; node_num++) {
		_current_node = node_num; 
		//Nodes
		for (pugi::xml_node node = doc.child("Simulation").child("Nodes").child("Node"); node; node = node.next_sibling("Node")) {
			std::string node_name = interpretValueAsString(std::string(node.attribute("name").value())) + std::string("_") + std::to_string(node_num);
			std::cout << "Found Node " << node_name << ".\n";

			// Check what type the node is
			MPILib::NodeType node_type = MPILib::NEUTRAL;
			if (std::string("EXCITATORY_DIRECT") == interpretValueAsString(std::string(node.attribute("type").value())))
				node_type = MPILib::EXCITATORY_DIRECT;
			if (std::string("INHIBITORY_DIRECT") == interpretValueAsString(std::string(node.attribute("type").value())))
				node_type = MPILib::INHIBITORY_DIRECT;
			if (std::string("INHIBITORY") == interpretValueAsString(std::string(node.attribute("type").value())))
				node_type = MPILib::INHIBITORY_DIRECT;
			if (std::string("EXCITATORY") == interpretValueAsString(std::string(node.attribute("type").value())))
				node_type = MPILib::EXCITATORY_DIRECT;
			// todo : Add gaussian node types when required.

			std::string algorithm_name = interpretValueAsString(std::string(node.attribute("algorithm").value()));

			MPILib::NodeId id = MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::network.addNode(*_algorithms[algorithm_name], node_type);
			_node_ids[node_name] = id;
		}

		//Connections
		for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("Connection"); conn; conn = conn.next_sibling("Connection")) {
			addConnection(conn);
		}

		//Incoming Connections
		for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("IncomingConnection"); conn; conn = conn.next_sibling("IncomingConnection")) {
			addIncomingConnection(conn);
		}

		//Outgoing Connections
		for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("OutgoingConnection"); conn; conn = conn.next_sibling("OutgoingConnection")) {
			std::string node = interpretValueAsString(std::string(conn.attribute("Node").value())) + std::string("_") + std::to_string(node_num);
			MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::network.setNodeExternalSuccessor(_node_ids[node]);
			_ordered_output_nodes.push_back(node);
		}

		//Reporting Densities
		for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Density"); conn; conn = conn.next_sibling("Density")) {
			std::string node = interpretValueAsString(std::string(conn.attribute("node").value())) + std::string("_") + std::to_string(node_num);
			double t_start = interpretValueAsDouble(std::string(conn.attribute("t_start").value()));
			double t_end = interpretValueAsDouble(std::string(conn.attribute("t_end").value()));
			double t_interval = interpretValueAsDouble(std::string(conn.attribute("t_interval").value()));

			_density_nodes.push_back(_node_ids[node]);
			_density_node_start_times.push_back(t_start);
			_density_node_end_times.push_back(t_end);
			_density_node_intervals.push_back(t_interval);
		}

		//Reporting Rates
		for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Rate"); conn; conn = conn.next_sibling("Rate")) {
			std::string node = interpretValueAsString(std::string(conn.attribute("node").value())) + std::string("_") + std::to_string(node_num);
			double t_interval = interpretValueAsDouble(std::string(conn.attribute("t_interval").value()));

			_rate_nodes.push_back(_node_ids[node]);
			_rate_node_intervals.push_back(t_interval);
		}

		//Reporting Display
		for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Display"); conn; conn = conn.next_sibling("Display")) {
			std::string node = interpretValueAsString(std::string(conn.attribute("node").value())) + std::string("_") + std::to_string(node_num);

			_display_nodes.push_back(_node_ids[node]);
		}
	}


	//Simulation Parameters
	double simulation_length = interpretValueAsDouble(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_end")));
	double time_step = interpretValueAsDouble(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_step")));
	std::string log_filename = interpretValueAsString(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("name_log")));

	MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_simulation_length = simulation_length;
	MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_time_step = time_step;

	MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::report_handler = new MPILib::report::handler::InactiveReportHandler();

	MPILib::SimulationRunParameter par_run(*MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::report_handler, (simulation_length / time_step) + 1, 0,
		simulation_length, time_step, time_step, log_filename);

	MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::network.configureSimulation(par_run);

}

template<class WeightType>
void SimulationParserCPU<WeightType>::startSimulation() {
	if (_display_nodes.size() > 0)
		TwoDLib::Display::getInstance()->animate(true, _display_nodes, MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_time_step);
	MPILib::MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::startSimulation();
}

template<class WeightType>
void SimulationParserCPU<WeightType>::init() {
	parseXmlFile();
}

template<class WeightType>
std::vector<double> SimulationParserCPU<WeightType>::evolveSingleStep(std::vector<double> activity) {

	MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::network.reportNodeActivities(_rate_nodes, _rate_node_intervals,
		(_count * MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_time_step));

	if (_display_nodes.size() > 0)
		TwoDLib::Display::getInstance()->updateDisplay(_count);

	TwoDLib::GridReport<WeightType>::getInstance()->reportDensity(_density_nodes, _density_node_start_times, _density_node_end_times, _density_node_intervals,
		(_count * MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_time_step));

	_count++;

	return MPILib::MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::evolveSingleStep(activity);
}

template<class WeightType>
bool SimulationParserCPU<WeightType>::simulationComplete() {
	return (_count * MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_time_step >=
		MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_simulation_length);
}