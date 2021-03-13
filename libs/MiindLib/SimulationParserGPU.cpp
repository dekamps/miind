#include "SimulationParserGPU.h"
#include <TwoDLib/XML.hpp>

template<>
SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(int num_nodes, const std::string xml_filename, std::map<std::string,std::string> vars) :
	SimulationParserCPU(num_nodes, xml_filename, vars), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(const std::string xml_filename, std::map<std::string, std::string> vars) :
	SimulationParserCPU(xml_filename, vars), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(int num_nodes, const std::string xml_filename) :
	SimulationParserCPU(num_nodes, xml_filename), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(const std::string xml_filename) :
	SimulationParserCPU(xml_filename), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::DelayedConnection>::SimulationParserGPU(int num_nodes, const std::string xml_filename, std::map<std::string, std::string> vars) :
	SimulationParserCPU(num_nodes, xml_filename, vars), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::DelayedConnection>::SimulationParserGPU(const std::string xml_filename, std::map<std::string, std::string> vars) :
	SimulationParserCPU(xml_filename, vars), vec_network(0.001) {
}


template<>
SimulationParserGPU<MPILib::DelayedConnection>::SimulationParserGPU(int num_nodes, const std::string xml_filename) :
	SimulationParserCPU(num_nodes, xml_filename), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::DelayedConnection>::SimulationParserGPU(const std::string xml_filename) :
	SimulationParserCPU(xml_filename), vec_network(0.001) {
}

template<class WeightType >
int SimulationParserGPU<WeightType>::interpretValueAsInt(std::string value) {

	if (value == "")
		return 0;

	if (SimulationParserCPU<WeightType>::_variables.find(value) == SimulationParserCPU<WeightType>::_variables.end()) // If the string isn't in the map, assume it's just a value
		return std::stoi(value);

	if (SimulationParserCPU<WeightType>::_variables[value] == "")
		std::cout << "Warning: The value of variable " << value << " in xml file is empty and cannot be converted to a number.\n";

	return std::stoi(SimulationParserCPU<WeightType>::_variables[value]);
}

template<class WeightType>
void SimulationParserGPU<WeightType>::endSimulation() {
	SimulationParserCPU<WeightType>::endSimulation();
}

template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addGridConnection(pugi::xml_node& xml_conn) {
	
	std::map<std::string, std::string> connection_parameters;

	std::string in = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::CustomConnectionParameters>::_current_node);
	std::string out = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::CustomConnectionParameters>::_current_node);

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("In") == std::string(ait->name())) || (std::string("Out") == std::string(ait->name())))
			continue;

		connection_parameters[std::string(ait->name())] = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	vec_network.addGridConnection(SimulationParserCPU<MPILib::CustomConnectionParameters>::_node_ids[in], SimulationParserCPU<MPILib::CustomConnectionParameters>::_node_ids[out], connection_parameters);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addGridConnection(pugi::xml_node& xml_conn) {

	std::string in = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::DelayedConnection>::_current_node);
	std::string out = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::DelayedConnection>::_current_node);

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	std::map<std::string, std::string> connection_parameters;
	connection_parameters["num_connections"] = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(num_connections));
	connection_parameters["efficacy"] = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(efficacy));
	connection_parameters["delay"] = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(delay));

	vec_network.addGridConnection(SimulationParserCPU<MPILib::DelayedConnection>::_node_ids[in], SimulationParserCPU<MPILib::DelayedConnection>::_node_ids[out], connection_parameters);
}

template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addMeshConnection(pugi::xml_node& xml_conn) {

	std::map<std::string, std::string> connection_parameters;

	std::string in = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::CustomConnectionParameters>::_current_node);
	std::string out = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::CustomConnectionParameters>::_current_node);
	std::string efficacy = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("efficacy").value()));

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("efficacy") == std::string(ait->name())) || (std::string("In") == std::string(ait->name()))
			|| (std::string("Out") == std::string(ait->name())))
			continue;

		connection_parameters[std::string(ait->name())] = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	vec_network.addMeshCustomConnection(SimulationParserCPU<MPILib::CustomConnectionParameters>::_node_ids[in], SimulationParserCPU<MPILib::CustomConnectionParameters>::_node_ids[out], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[out]][SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsDouble(efficacy)]);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addMeshConnection(pugi::xml_node& xml_conn) {

	std::string in = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(xml_conn.attribute("In").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::DelayedConnection>::_current_node);
	std::string out = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(xml_conn.attribute("Out").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::DelayedConnection>::_current_node);

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	vec_network.addMeshConnection(SimulationParserCPU<MPILib::DelayedConnection>::_node_ids[in],
		SimulationParserCPU<MPILib::DelayedConnection>::_node_ids[out],
		interpretValueAsDouble(std::string(efficacy)),
		interpretValueAsDouble(std::string(num_connections)),
		interpretValueAsDouble(std::string(delay)),
		&_mesh_transition_matrics[_node_algorithm_mapping[out]][SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsDouble(std::string(efficacy))]);
}

template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addIncomingGridConnection(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string node = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::CustomConnectionParameters>::_current_node);

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection_parameters[std::string(ait->name())] = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
	vec_network.addGridConnection(SimulationParserCPU<MPILib::CustomConnectionParameters>::_node_ids[node], connection_parameters, _external_node_count);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addIncomingGridConnection(pugi::xml_node& xml_conn) {
	std::string node = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::DelayedConnection>::_current_node);

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	std::map<std::string, std::string> connection_parameters;
	connection_parameters["num_connections"] = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(num_connections));
	connection_parameters["efficacy"] = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(efficacy));
	connection_parameters["delay"] = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(delay));
	
	vec_network.addGridConnection(SimulationParserCPU<MPILib::DelayedConnection>::_node_ids[node], connection_parameters, _external_node_count);
}


template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addIncomingMeshConnection(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string node = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::CustomConnectionParameters>::_current_node);
	std::string efficacy = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(xml_conn.attribute("efficacy").value()));

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection_parameters[std::string(ait->name())] = SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
		
	vec_network.addMeshCustomConnection(SimulationParserCPU<MPILib::CustomConnectionParameters>::_node_ids[node], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[node]][SimulationParserCPU<MPILib::CustomConnectionParameters>::interpretValueAsDouble(efficacy)], _external_node_count);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addIncomingMeshConnection(pugi::xml_node& xml_conn) {
	std::string node = SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsString(std::string(xml_conn.attribute("Node").value())) + std::string("_") + std::to_string(SimulationParserCPU<MPILib::DelayedConnection>::_current_node);
	
	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	vec_network.addMeshConnection(SimulationParserCPU<MPILib::DelayedConnection>::_node_ids[node], 
		interpretValueAsDouble(std::string(efficacy)), 
		interpretValueAsDouble(std::string(num_connections)), 
		interpretValueAsDouble(std::string(delay)), 
		&_mesh_transition_matrics[_node_algorithm_mapping[node]][SimulationParserCPU<MPILib::DelayedConnection>::interpretValueAsDouble(std::string(efficacy))], 
		_external_node_count);
}

template<class WeightType>
bool SimulationParserGPU<WeightType>::addGridAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("GridAlgorithmGroup") == SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("name").value()));
			if (alg_name != algorithm_name)
				continue;

			std::string model_filename = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.attribute("tau_refractive").value()));
			std::string transform_filename = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("transformfile").value()));
			double start_v = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.attribute("start_v").value()));
			double start_w = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.attribute("start_w").value()));
			double time_step = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));

			// todo: Check time_step matches network time step

			pugi::xml_parse_result model_file_xml = doc.load_file(model_filename.c_str());
			pugi::xml_node model = doc.first_child();

			_meshes.push_back(TwoDLib::RetrieveMeshFromXML(model));
			_reversal_mappings.push_back(TwoDLib::RetrieveMappingFromXML("Reversal", model));
			_reset_mappings.push_back(TwoDLib::RetrieveMappingFromXML("Reset", model));
			_transition_mats.push_back(TwoDLib::TransitionMatrix(transform_filename));

			vec_network.addGridNode(_meshes.back(), _transition_mats.back(), start_v, start_w, _reversal_mappings.back(), _reset_mappings.back(), tau_refractive);
			return true;
		}
	}
	return false;
}

template<class WeightType>
bool SimulationParserGPU<WeightType>::addMeshAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("MeshAlgorithmGroup") == SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("name").value()));
			if (alg_name != algorithm_name)
				continue;

			std::string model_filename = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.attribute("tau_refractive").value()));
			double time_step = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));

			// Only load the matrices once for each algorithm.
			if (!_mesh_transition_matrics.count(algorithm_name)) {
				std::map<double, TwoDLib::TransitionMatrix> matrices;
				for (pugi::xml_node matrix_file = algorithm.child("MatrixFile"); matrix_file; matrix_file = matrix_file.next_sibling("MatrixFile")) {
					// In this version, lets say that the efficacy must match the file name of the associated matrix file - makes so much more sense
					// than to quote the efficacy value and hope we choose the correct mat file
					auto s = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(matrix_file.child_value()));
					auto tm = TwoDLib::TransitionMatrix(s);
					matrices[tm.Efficacy()] = tm;
				}

				_mesh_transition_matrics[algorithm_name] = matrices;
			}

			// todo: Check time_step matches network time step

			pugi::xml_parse_result model_file_xml = doc.load_file(model_filename.c_str());
			pugi::xml_node model = doc.first_child();

			_meshes.push_back(TwoDLib::RetrieveMeshFromXML(model));
			_reversal_mappings.push_back(TwoDLib::RetrieveMappingFromXML("Reversal", model));
			_reset_mappings.push_back(TwoDLib::RetrieveMappingFromXML("Reset", model));

			vec_network.addMeshNode(_meshes.back(), _reversal_mappings.back(), _reset_mappings.back(), tau_refractive);

			return true;
		}
	}
	return false;
}

template<class WeightType>
bool SimulationParserGPU<WeightType>::addRateFunctorNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("RateFunctor") == SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just assume a constant rate for now
			std::string algorithm_name = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(algorithm.attribute("name").value()));
			if (alg_name != algorithm_name)
				continue;

			double rate = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(algorithm.child_value("expression")));

			rate_functor rf(rate);
			_rate_functors.push_back(rf);

			vec_network.addRateNode(_rate_functors.back());
			return true;
		}
	}
	return false;
}

template<class WeightType>
void SimulationParserGPU<WeightType>::parseXmlFile() {
	pugi::xml_document doc;
	if (!doc.load_file(SimulationParserCPU<WeightType>::_xml_filename.c_str()))
		std::cout << "Failed to load XML simulation file.\n";

	//check Weight Type matches this class
	if (!SimulationParserCPU<WeightType>::checkWeightType(doc))
		return;

	// Load Variables into map
	for (pugi::xml_node var = doc.child("Simulation").child("Variable"); var; var = var.next_sibling("Variable")) {
		if (!SimulationParserCPU<WeightType>::_variables.count(std::string(var.attribute("Name").value())))
			SimulationParserCPU<WeightType>::_variables[std::string(var.attribute("Name").value())] = std::string(var.text().as_string());
	}

	//Algorithms - In the CUDA version we don't store the algorithm, just search for the correct algorithm to add each node
	// In the Cuda version, we just deal with "Group" algorithms and RateFunctor

	SimulationParserCPU<WeightType>::_node_ids = std::map<std::string, MPILib::NodeId>();
	_external_node_count = 0;

	for (unsigned int node_num = 0; node_num < MiindTvbModelAbstract<WeightType, MPILib::utilities::CircularDistribution>::_num_nodes; node_num++) {
		SimulationParserCPU<WeightType>::_current_node = node_num;

		//Nodes
		for (pugi::xml_node node = doc.child("Simulation").child("Nodes").child("Node"); node; node = node.next_sibling("Node")) {
			std::string node_name = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(node.attribute("name").value())) + std::string("_") + std::to_string(node_num);
			std::cout << "Found Node " << node_name << ".\n";

			// Check what type the node is
			/* In the CUDA version, there's currently no check for correct NodeType
			*/

			std::string algorithm_name = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(node.attribute("algorithm").value()));

			pugi::xml_document check_doc;
			check_doc.load_file(SimulationParserCPU<WeightType>::_xml_filename.c_str());

			_node_algorithm_mapping[node_name] = algorithm_name;

			// We just call all of the add functions here - the algorithm name will only match one algorithm in the correct function.
			if (addGridAlgorithmGroupNode(check_doc, algorithm_name))
				_node_algorithm_types[node_name] = std::string("grid");
			if (addMeshAlgorithmGroupNode(check_doc, algorithm_name))
				_node_algorithm_types[node_name] = std::string("mesh");
			if (addRateFunctorNode(check_doc, algorithm_name))
				_node_algorithm_types[node_name] = std::string("rate");

			MPILib::NodeId test = SimulationParserCPU<WeightType>::_node_ids.size();
			SimulationParserCPU<WeightType>::_node_ids[node_name] = test;
		}

		//Connections
		for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("Connection"); conn; conn = conn.next_sibling("Connection")) {
			std::string conn_out = std::string(conn.attribute("Out").value()) + std::string("_") + std::to_string(node_num);
			if (_node_algorithm_types[conn_out] == std::string("grid"))
				addGridConnection(conn);
			if (_node_algorithm_types[conn_out] == std::string("mesh"))
				addMeshConnection(conn);
		}

		//Incoming Connections
		for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("IncomingConnection"); conn; conn = conn.next_sibling("IncomingConnection")) {
			std::string conn_node = std::string(conn.attribute("Node").value()) + std::string("_") + std::to_string(node_num);
			if (_node_algorithm_types[conn_node] == std::string("grid"))
				addIncomingGridConnection(conn);
			if (_node_algorithm_types[conn_node] == std::string("mesh"))
				addIncomingMeshConnection(conn);

			_external_node_count++;
		}

		//Outgoing Connections
		for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("OutgoingConnection"); conn; conn = conn.next_sibling("OutgoingConnection")) {
			std::string node = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(conn.attribute("Node").value())) + std::string("_") + std::to_string(node_num);
			vec_network.addExternalMonitor(SimulationParserCPU<WeightType>::_node_ids[node]);
		}

		//Reporting Densities
		for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Density"); conn; conn = conn.next_sibling("Density")) {
			std::string node = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(conn.attribute("node").value())) + std::string("_") + std::to_string(node_num);
			double t_start = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(conn.attribute("t_start").value()));
			double t_end = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(conn.attribute("t_end").value()));
			double t_interval = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(conn.attribute("t_interval").value()));

			SimulationParserCPU<WeightType>::_density_nodes.push_back(SimulationParserCPU<WeightType>::_node_ids[node]);
			SimulationParserCPU<WeightType>::_density_node_start_times.push_back(t_start);
			SimulationParserCPU<WeightType>::_density_node_end_times.push_back(t_end);
			SimulationParserCPU<WeightType>::_density_node_intervals.push_back(t_interval);
		}

		//Reporting Rates
		for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Rate"); conn; conn = conn.next_sibling("Rate")) {
			std::string node = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(conn.attribute("node").value())) + std::string("_") + std::to_string(node_num);
			double t_interval = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(conn.attribute("t_interval").value()));

			SimulationParserCPU<WeightType>::_rate_nodes.push_back(SimulationParserCPU<WeightType>::_node_ids[node]);
			SimulationParserCPU<WeightType>::_rate_node_intervals.push_back(t_interval);
		}

		//Reporting Display
		for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Display"); conn; conn = conn.next_sibling("Display")) {
			std::string node = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(conn.attribute("node").value())) + std::string("_") + std::to_string(node_num);

			SimulationParserCPU<WeightType>::_display_nodes.push_back(SimulationParserCPU<WeightType>::_node_ids[node]);
		}
	}


	//Simulation Parameters
	double simulation_length = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_end")));
	double time_step = SimulationParserCPU<WeightType>::interpretValueAsDouble(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_step")));
	std::string log_filename = SimulationParserCPU<WeightType>::interpretValueAsString(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("name_log")));
	unsigned int master_steps = interpretValueAsInt(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("master_steps")));

	if (master_steps == 0) // Let's assume that if we didn't include a master_steps or we set it to 0, we just want the default of 10.
		master_steps = 10;

	SimulationParserCPU<WeightType>::_simulation_length = simulation_length;
	SimulationParserCPU<WeightType>::_time_step = time_step;
	vec_network.setTimeStep(time_step);

	vec_network.setDisplayNodes(SimulationParserCPU<WeightType>::_display_nodes);
	vec_network.setRateNodes(SimulationParserCPU<WeightType>::_rate_nodes, SimulationParserCPU<WeightType>::_rate_node_intervals);
	vec_network.setDensityNodes(SimulationParserCPU<WeightType>::_density_nodes, SimulationParserCPU<WeightType>::_density_node_start_times, SimulationParserCPU<WeightType>::_density_node_end_times, SimulationParserCPU<WeightType>::_density_node_intervals);

	vec_network.initOde2DSystem(master_steps);

}

template<class WeightType>
void SimulationParserGPU<WeightType>::startSimulation() {
	vec_network.setupLoop(true);
	SimulationParserCPU<WeightType>::pb = new MPILib::utilities::ProgressBar((int)(SimulationParserCPU<WeightType>::_simulation_length / SimulationParserCPU<WeightType>::_time_step));
}

template<class WeightType>
void SimulationParserGPU<WeightType>::init() {
	parseXmlFile();
}

template<class WeightType>
std::vector<double> SimulationParserGPU<WeightType>::evolveSingleStep(std::vector<double> activity) {
	std::vector<double> out_activities;
	for(auto& it : vec_network.singleStep(activity, SimulationParserCPU<WeightType>::_count)) {
		out_activities.push_back(it);
	}

	SimulationParserCPU<WeightType>::_count++;
	(*SimulationParserCPU<WeightType>::pb)++;

	return out_activities;
}

template<class WeightType>
bool SimulationParserGPU<WeightType>::simulationComplete() {
	return (SimulationParserCPU<WeightType>::_count * SimulationParserCPU<WeightType>::_time_step >=
		SimulationParserCPU<WeightType>::_simulation_length);
}