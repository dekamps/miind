#include "SimulationParserGPU.h"
#include <TwoDLib\XML.hpp>

SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(int num_nodes, const std::string xml_filename) :
	// For now we don't allow num_nodes : override to 1 node only.
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>(1, 1.0), vec_network(0.001), _count(0), _xml_filename(xml_filename) {
}

SimulationParserGPU<MPILib::CustomConnectionParameters>::SimulationParserGPU(const std::string xml_filename) :
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>(1, 1.0), vec_network(0.001), _count(0), _xml_filename(xml_filename) {
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::endSimulation() {
	MPILib::MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::endSimulation();
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::addGridConnectionCCP(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string in = std::string(xml_conn.attribute("In").value());
	std::string out = std::string(xml_conn.attribute("Out").value());

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("In") == std::string(ait->name())) || (std::string("Out") == std::string(ait->name())))
			continue;

		connection_parameters[std::string(ait->name())] = std::string(ait->value());
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	vec_network.addGridConnection(_node_ids[in], _node_ids[out], connection_parameters);
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::addMeshConnectionCCP(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string in = std::string(xml_conn.attribute("In").value());
	std::string out = std::string(xml_conn.attribute("Out").value());
	std::string efficacy = std::string(xml_conn.attribute("efficacy").value());

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("efficacy") == std::string(ait->name())) || (std::string("In") == std::string(ait->name()))
			|| (std::string("Out") == std::string(ait->name())))
			continue;

		connection_parameters[std::string(ait->name())] = std::string(ait->value());
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	vec_network.addMeshCustomConnection(_node_ids[in], _node_ids[out], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[out]][efficacy]);
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::addIncomingGridConnectionCCP(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string node = std::string(xml_conn.attribute("Node").value());

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection_parameters[std::string(ait->name())] = std::string(ait->value());
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
	vec_network.addGridConnection(_node_ids[node], connection_parameters, _external_node_count);
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::addIncomingMeshConnectionCCP(pugi::xml_node& xml_conn) {
	std::map<std::string, std::string> connection_parameters;

	std::string node = std::string(xml_conn.attribute("Node").value());
	std::string efficacy = std::string(xml_conn.attribute("efficacy").value());

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection_parameters[std::string(ait->name())] = std::string(ait->value());
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
		
	vec_network.addMeshCustomConnection(_node_ids[node], connection_parameters, &_mesh_transition_matrics[_node_algorithm_mapping[node]][efficacy], _external_node_count);
}

bool SimulationParserGPU<MPILib::CustomConnectionParameters>::addGridAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("GridAlgorithmGroup") == std::string(algorithm.attribute("type").value())) {
			std::string algorithm_name = std::string(algorithm.attribute("name").value());
			if (alg_name != algorithm_name)
				continue;

			std::string model_filename = std::string(algorithm.attribute("modelfile").value());
			double tau_refractive = std::stod(std::string(algorithm.attribute("tau_refractive").value()));
			std::string transform_filename = std::string(algorithm.attribute("transformfile").value());
			double start_v = std::stod(std::string(algorithm.attribute("start_v").value()));
			double start_w = std::stod(std::string(algorithm.attribute("start_w").value()));
			double time_step = std::stod(std::string(algorithm.child_value("TimeStep")));

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

bool SimulationParserGPU<MPILib::CustomConnectionParameters>::addMeshAlgorithmGroupNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("MeshAlgorithmGroup") == std::string(algorithm.attribute("type").value())) {
			std::string algorithm_name = std::string(algorithm.attribute("name").value());
			if (alg_name != algorithm_name)
				continue;

			std::string model_filename = std::string(algorithm.attribute("modelfile").value());
			double tau_refractive = std::stod(std::string(algorithm.attribute("tau_refractive").value()));
			double time_step = std::stod(std::string(algorithm.child_value("TimeStep")));

			std::map<std::string, TwoDLib::TransitionMatrix> matrices;
			for (pugi::xml_node matrix_file = algorithm.child("MatrixFile"); matrix_file; matrix_file = matrix_file.next_sibling("MatrixFile")) {
				// In this version, lets say that the efficacy must match the file name of the associated matrix file - makes so much more sense
				// than to quote the efficacy value and hope we choose the correct mat file
				auto s = std::string(matrix_file.child_value());
				matrices[s] = TwoDLib::TransitionMatrix(s);
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

bool SimulationParserGPU<MPILib::CustomConnectionParameters>::addRateFunctorNode(pugi::xml_document& doc, std::string alg_name) {
	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		if (std::string("RateFunctor") == std::string(algorithm.attribute("type").value())) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just assume a constant rate for now
			std::string algorithm_name = std::string(algorithm.attribute("name").value());
			if (alg_name != algorithm_name)
				continue;

			double rate = std::stod(std::string(algorithm.child_value("expression")));

			rate_functor rf(rate);
			_rate_functors.push_back(rf);

			vec_network.addRateNode(_rate_functors.back());
			return true;
		}
	}
	return false;
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::parseXmlFile() {
	pugi::xml_document doc;
	if (!doc.load_file(_xml_filename.c_str()))
		std::cout << "Failed to load XML simulation file.\n";

	//check Weight Type matches this class
	if (std::string("CustomConnectionParameters") != std::string(doc.child("Simulation").child_value("WeightType"))) {
		std::cout << "The weight type of the SimulationParser (" << "CustomConnectionParameters" << ") doesn't match the WeightType in the XML file (" << doc.child("Simulation").child_value("WeightType") << "). Exiting.\n";
		return;
	}

	//Algorithms - In the CUDA version we don't store the algorithm, just search for the correct algorithm to add each node
	// In the Cuda version, we just deal with "Group" algorithms and RateFunctor

	_node_ids = std::map<std::string, MPILib::NodeId>();
	_external_node_count = 0;
	//Nodes
	for (pugi::xml_node node = doc.child("Simulation").child("Nodes").child("Node"); node; node = node.next_sibling("Node")) {
		std::string node_name = std::string(node.attribute("name").value());
		std::cout << "Found Node " << node_name << ".\n";
			
		// Check what type the node is
		/* In the CUDA version, there's currently no check for correct NodeType
		MPILib::NodeType node_type = MPILib::NEUTRAL;
		if (std::string("EXCITATORY_DIRECT") == std::string(node.attribute("type").value()))
			node_type = MPILib::EXCITATORY_DIRECT;
		if (std::string("INHIBITORY_DIRECT") == std::string(node.attribute("type").value()))
			node_type = MPILib::INHIBITORY_DIRECT;
		if (std::string("INHIBITORY") == std::string(node.attribute("type").value()))
			node_type = MPILib::INHIBITORY_DIRECT;
		if (std::string("EXCITATORY") == std::string(node.attribute("type").value()))
			node_type = MPILib::EXCITATORY_DIRECT;
		// todo : Add gaussian node types when required.
		*/

		std::string algorithm_name = std::string(node.attribute("algorithm").value());

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
		// A better way to do this is to move the connection building to a separate concrete non-templated class
		// too lazy right now...
		std::string conn_out = std::string(conn.attribute("Out").value());
		if (_node_algorithm_types[conn_out] == std::string("grid"))
			addGridConnectionCCP(conn);
		if (_node_algorithm_types[conn_out] == std::string("mesh"))
			addMeshConnectionCCP(conn);
		// todo : Deal with other connection types - DelayedConnection, double
	}

	//Incoming Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("IncomingConnection"); conn; conn = conn.next_sibling("IncomingConnection")) {
		// A better way to do this is to move the connection building to a separate concrete non-templated class
		// too lazy right now...
		std::string conn_node = std::string(conn.attribute("Node").value());
		if (_node_algorithm_types[conn_node] == std::string("grid"))
			addIncomingGridConnectionCCP(conn);
		if (_node_algorithm_types[conn_node] == std::string("mesh"))
			addIncomingMeshConnectionCCP(conn);

		_external_node_count++;
		// todo : Deal with other connection types - DelayedConnection, double
	}

	//Outgoing Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("OutgoingConnection"); conn; conn = conn.next_sibling("OutgoingConnection")) {
		std::string node = std::string(conn.attribute("Node").value());
		vec_network.addExternalMonitor(_node_ids[node]);
	}

	//Reporting Densities
	for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Density"); conn; conn = conn.next_sibling("Density")) {
		std::string node = std::string(conn.attribute("node").value());
		double t_start = std::stod(std::string(conn.attribute("t_start").value()));
		double t_end = std::stod(std::string(conn.attribute("t_end").value()));
		double t_interval = std::stod(std::string(conn.attribute("t_interval").value()));

		_density_nodes.push_back(_node_ids[node]);
		_density_node_start_times.push_back(t_start);
		_density_node_end_times.push_back(t_end);
		_density_node_intervals.push_back(t_interval);
	}

	//Reporting Rates
	for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Rate"); conn; conn = conn.next_sibling("Rate")) {
		std::string node = std::string(conn.attribute("node").value());
		double t_interval = std::stod(std::string(conn.attribute("t_interval").value()));

		_rate_nodes.push_back(_node_ids[node]);
		_rate_node_intervals.push_back(t_interval);
	}

	//Reporting Display
	for (pugi::xml_node conn = doc.child("Simulation").child("Reporting").child("Display"); conn; conn = conn.next_sibling("Display")) {
		std::string node = std::string(conn.attribute("node").value());

		_display_nodes.push_back(_node_ids[node]);
	}


	//Simulation Parameters
	double simulation_length = std::stod(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_end")));
	double time_step = std::stod(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("t_step")));
	std::string log_filename = std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("name_log"));
	unsigned int master_steps = std::stoi(std::string(doc.child("Simulation").child("SimulationRunParameter").child_value("master_steps")));

	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_simulation_length = simulation_length;
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step = time_step;
	vec_network.setTimeStep(time_step);

	vec_network.setDisplayNodes(_display_nodes);
	vec_network.setRateNodes(_rate_nodes, _rate_node_intervals);
	vec_network.setDensityNodes(_density_nodes, _density_node_start_times, _density_node_end_times, _density_node_intervals);

	vec_network.initOde2DSystem(master_steps);

}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::startSimulation() {
	vec_network.setupLoop(true);
	pb = new MPILib::utilities::ProgressBar((int)(MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_simulation_length / MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step));
}

void SimulationParserGPU<MPILib::CustomConnectionParameters>::init() {
	parseXmlFile();
}

std::vector<double> SimulationParserGPU<MPILib::CustomConnectionParameters>::evolveSingleStep(std::vector<double> activity) {
	std::vector<double> out_activities;
	for(auto& it : vec_network.singleStep(activity,_count)) {
		out_activities.push_back(it);
	}

	_count++;
	(*pb)++;

	return out_activities;
}

bool SimulationParserGPU<MPILib::CustomConnectionParameters>::simulationComplete() {
	return (_count * MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step >=
		MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_simulation_length);
}