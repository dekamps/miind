#include "SimulationParserGPU.h"
#include <TwoDLib\XML.hpp>

template<>
SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(int num_nodes, const std::string xml_filename) :
	// For now we don't allow num_nodes : override to 1 node only.
	SimulationParserCPU(num_nodes, xml_filename), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(const std::string xml_filename) :
	SimulationParserCPU(1, xml_filename), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::DelayedConnection>::SimulationParserGPU(int num_nodes, const std::string xml_filename) :
	// For now we don't allow num_nodes : override to 1 node only.
	SimulationParserCPU(num_nodes, xml_filename), vec_network(0.001) {
}

template<>
SimulationParserGPU<MPILib::DelayedConnection>::SimulationParserGPU(const std::string xml_filename) :
	SimulationParserCPU(1, xml_filename), vec_network(0.001) {
}

template<class WeightType >
int SimulationParserGPU<WeightType>::interpretValueAsInt(std::string value) {

	if (value == "")
		return 0;

	if (_variables.find(value) == _variables.end()) // If the string isn't in the map, assume it's just a value
		return std::stoi(value);

	if (_variables[value] == "")
		std::cout << "Warning: The value of variable " << value << " in xml file is empty and cannot be converted to a number.\n";

	return std::stoi(_variables[value]);
}

template<class WeightType>
void SimulationParserGPU<WeightType>::endSimulation() {
	SimulationParserCPU::endSimulation();
}

template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addGridConnection(pugi::xml_node& xml_conn) {
	
	std::map<std::string, std::string> connection_parameters;

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value()));
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value()));

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("In") == std::string(ait->name())) || (std::string("Out") == std::string(ait->name())))
			continue;

		connection_parameters[std::string(ait->name())] = interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	vec_network.addGridConnection(_node_ids[in], _node_ids[out], connection_parameters);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addGridConnection(pugi::xml_node& xml_conn) {

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value()));
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value()));

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	std::map<std::string, std::string> connection_parameters;
	connection_parameters["num_connections"] = interpretValueAsString(std::string(num_connections));
	connection_parameters["efficacy"] = interpretValueAsString(std::string(efficacy));
	connection_parameters["delay"] = interpretValueAsString(std::string(delay));

	vec_network.addGridConnection(_node_ids[in], _node_ids[out], connection_parameters);
}

template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addMeshConnection(pugi::xml_node& xml_conn) {

	std::map<std::string, std::string> connection_parameters;

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value()));
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value()));
	std::string efficacy = interpretValueAsString(std::string(xml_conn.attribute("efficacy").value()));

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("efficacy") == std::string(ait->name())) || (std::string("In") == std::string(ait->name()))
			|| (std::string("Out") == std::string(ait->name())))
			continue;

		connection_parameters[std::string(ait->name())] = interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	vec_network.addMeshCustomConnection(_node_ids[in], _node_ids[out], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[out]][interpretValueAsDouble(efficacy)]);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addMeshConnection(pugi::xml_node& xml_conn) {

	std::string in = interpretValueAsString(std::string(xml_conn.attribute("In").value()));
	std::string out = interpretValueAsString(std::string(xml_conn.attribute("Out").value()));

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	std::map<std::string, std::string> connection_parameters;
	connection_parameters["num_connections"] = interpretValueAsString(std::string(num_connections));
	connection_parameters["efficacy"] = interpretValueAsString(std::string(efficacy));
	connection_parameters["delay"] = interpretValueAsString(std::string(delay));

	vec_network.addMeshCustomConnection(_node_ids[in], _node_ids[out], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[out]][interpretValueAsDouble(std::string(efficacy))]);
}

template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addIncomingGridConnection(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value()));

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection_parameters[std::string(ait->name())] = interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
	vec_network.addGridConnection(_node_ids[node], connection_parameters, _external_node_count);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addIncomingGridConnection(pugi::xml_node& xml_conn) {
	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value()));

	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	std::map<std::string, std::string> connection_parameters;
	connection_parameters["num_connections"] = interpretValueAsString(std::string(num_connections));
	connection_parameters["efficacy"] = interpretValueAsString(std::string(efficacy));
	connection_parameters["delay"] = interpretValueAsString(std::string(delay));
	
	vec_network.addGridConnection(_node_ids[node], connection_parameters, _external_node_count);
}


template<>
void SimulationParserGPU<MPILib::CustomConnectionParameters>::addIncomingMeshConnection(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value()));
	std::string efficacy = interpretValueAsString(std::string(xml_conn.attribute("efficacy").value()));

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection_parameters[std::string(ait->name())] = interpretValueAsString(std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
		
	vec_network.addMeshCustomConnection(_node_ids[node], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[node]][interpretValueAsDouble(efficacy)], _external_node_count);
}

template<>
void SimulationParserGPU<MPILib::DelayedConnection>::addIncomingMeshConnection(pugi::xml_node& xml_conn) {
	std::string node = interpretValueAsString(std::string(xml_conn.attribute("Node").value()));
	
	std::string values = std::string(xml_conn.text().as_string());
	char num_connections[255];
	char efficacy[255];
	char delay[255];
	std::sscanf(values.c_str(), "%s %s %s", num_connections, efficacy, delay);

	std::map<std::string, std::string> connection_parameters;
	connection_parameters["num_connections"] = interpretValueAsString(std::string(num_connections));
	connection_parameters["efficacy"] = interpretValueAsString(std::string(efficacy));
	connection_parameters["delay"] = interpretValueAsString(std::string(delay));

	vec_network.addMeshCustomConnection(_node_ids[node], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[node]][interpretValueAsDouble(std::string(efficacy))], _external_node_count);
}

template<class WeightType>
bool SimulationParserGPU<WeightType>::addGridAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("GridAlgorithmGroup") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			if (alg_name != algorithm_name)
				continue;

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(std::string(algorithm.attribute("tau_refractive").value()));
			std::string transform_filename = interpretValueAsString(std::string(algorithm.attribute("transformfile").value()));
			double start_v = interpretValueAsDouble(std::string(algorithm.attribute("start_v").value()));
			double start_w = interpretValueAsDouble(std::string(algorithm.attribute("start_w").value()));
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));

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
		if (std::string("MeshAlgorithmGroup") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			if (alg_name != algorithm_name)
				continue;

			std::string model_filename = interpretValueAsString(std::string(algorithm.attribute("modelfile").value()));
			double tau_refractive = interpretValueAsDouble(std::string(algorithm.attribute("tau_refractive").value()));
			double time_step = interpretValueAsDouble(std::string(algorithm.child_value("TimeStep")));

			std::map<double, TwoDLib::TransitionMatrix> matrices;
			for (pugi::xml_node matrix_file = algorithm.child("MatrixFile"); matrix_file; matrix_file = matrix_file.next_sibling("MatrixFile")) {
				// In this version, lets say that the efficacy must match the file name of the associated matrix file - makes so much more sense
				// than to quote the efficacy value and hope we choose the correct mat file
				auto s = interpretValueAsString(std::string(matrix_file.child_value()));
				auto tm = TwoDLib::TransitionMatrix(s);
				matrices[tm.Efficacy()] = tm;
			}

			_mesh_transition_matrics[algorithm_name] = matrices;

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
		if (std::string("RateFunctor") == interpretValueAsString(std::string(algorithm.attribute("type").value()))) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just assume a constant rate for now
			std::string algorithm_name = interpretValueAsString(std::string(algorithm.attribute("name").value()));
			if (alg_name != algorithm_name)
				continue;

			double rate = interpretValueAsDouble(std::string(algorithm.child_value("expression")));

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
	if (!doc.load_file(_xml_filename.c_str()))
		std::cout << "Failed to load XML simulation file.\n";

	//check Weight Type matches this class
	if (!checkWeightType(doc))
		return;

	// Load Variables into map
	for (pugi::xml_node var = doc.child("Simulation").child("Variable"); var; var = var.next_sibling("Variable")) {
		_variables[std::string(var.attribute("Name").value())] = std::string(var.text().as_string());
	}

	//Algorithms - In the CUDA version we don't store the algorithm, just search for the correct algorithm to add each node
	// In the Cuda version, we just deal with "Group" algorithms and RateFunctor

	_node_ids = std::map<std::string, MPILib::NodeId>();
	_external_node_count = 0;
	//Nodes
	for (pugi::xml_node node = doc.child("Simulation").child("Nodes").child("Node"); node; node = node.next_sibling("Node")) {
		std::string node_name = interpretValueAsString(std::string(node.attribute("name").value()));
		std::cout << "Found Node " << node_name << ".\n";
			
		// Check what type the node is
		/* In the CUDA version, there's currently no check for correct NodeType
		*/

		std::string algorithm_name = interpretValueAsString(std::string(node.attribute("algorithm").value()));

		pugi::xml_document check_doc;
		check_doc.load_file(_xml_filename.c_str());

		_node_algorithm_mapping[node_name] = algorithm_name;

		// We just call all of the add functions here - the algorithm name will only match one algorithm in the correct function.
		if (addGridAlgorithmGroupNode(check_doc, algorithm_name))
			_node_algorithm_types[node_name] = std::string("grid");
		if (addMeshAlgorithmGroupNode(check_doc, algorithm_name))
			_node_algorithm_types[node_name] = std::string("mesh");
		if (addRateFunctorNode(check_doc,algorithm_name))
			_node_algorithm_types[node_name] = std::string("rate");

		MPILib::NodeId test = _node_ids.size();
		_node_ids[node_name] = test;
	}

	//Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("Connection"); conn; conn = conn.next_sibling("Connection")) {
		std::string conn_out = std::string(conn.attribute("Out").value());
		if (_node_algorithm_types[conn_out] == std::string("grid"))
			addGridConnection(conn);
		if (_node_algorithm_types[conn_out] == std::string("mesh"))
			addMeshConnection(conn);
	}

	//Incoming Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("IncomingConnection"); conn; conn = conn.next_sibling("IncomingConnection")) {
		std::string conn_node = std::string(conn.attribute("Node").value());
		if (_node_algorithm_types[conn_node] == std::string("grid"))
			addIncomingGridConnection(conn);
		if (_node_algorithm_types[conn_node] == std::string("mesh"))
			addIncomingMeshConnection(conn);

		_external_node_count++;
	}

	//Outgoing Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("OutgoingConnection"); conn; conn = conn.next_sibling("OutgoingConnection")) {
		std::string node = interpretValueAsString(std::string(conn.attribute("Node").value()));
		vec_network.addExternalMonitor(_node_ids[node]);
	}

	//Reporting Densities
	for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Density"); conn; conn = conn.next_sibling("Density")) {
		std::string node = interpretValueAsString(std::string(conn.attribute("node").value()));
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
		std::string node = interpretValueAsString(std::string(conn.attribute("node").value()));
		double t_interval = interpretValueAsDouble(std::string(conn.attribute("t_interval").value()));

		_rate_nodes.push_back(_node_ids[node]);
		_rate_node_intervals.push_back(t_interval);
	}

	//Reporting Display
	for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Display"); conn; conn = conn.next_sibling("Display")) {
		std::string node = interpretValueAsString(std::string(conn.attribute("node").value()));

		_display_nodes.push_back(_node_ids[node]);
	}


	//Simulation Parameters
	double simulation_length = interpretValueAsDouble(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_end")));
	double time_step = interpretValueAsDouble(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_step")));
	std::string log_filename = interpretValueAsString(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("name_log")));
	unsigned int master_steps = interpretValueAsInt(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("master_steps")));

	SimulationParserCPU<WeightType>::_simulation_length = simulation_length;
	SimulationParserCPU<WeightType>::_time_step = time_step;
	vec_network.setTimeStep(time_step);

	vec_network.setDisplayNodes(_display_nodes);
	vec_network.setRateNodes(_rate_nodes, _rate_node_intervals);
	vec_network.setDensityNodes(_density_nodes, _density_node_start_times, _density_node_end_times, _density_node_intervals);

	vec_network.initOde2DSystem(master_steps);

}

template<class WeightType>
void SimulationParserGPU<WeightType>::startSimulation() {
	vec_network.setupLoop(true);
	pb = new MPILib::utilities::ProgressBar((int)(SimulationParserCPU<WeightType>::_simulation_length / SimulationParserCPU<WeightType>::_time_step));
}

template<class WeightType>
void SimulationParserGPU<WeightType>::init() {
	parseXmlFile();
}

template<class WeightType>
std::vector<double> SimulationParserGPU<WeightType>::evolveSingleStep(std::vector<double> activity) {
	std::vector<double> out_activities;
	for(auto& it : vec_network.singleStep(activity,_count)) {
		out_activities.push_back(it);
	}

	_count++;
	(*pb)++;

	return out_activities;
}

template<class WeightType>
bool SimulationParserGPU<WeightType>::simulationComplete() {
	return (_count * SimulationParserCPU<WeightType>::_time_step >=
		SimulationParserCPU<WeightType>::_simulation_length);
}