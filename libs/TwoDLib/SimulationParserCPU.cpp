#include "SimulationParserCPU.h"
#include <map>
#include <memory>
#include <TwoDLib/GridReport.hpp>

SimulationParserCPU<MPILib::CustomConnectionParameters>::SimulationParserCPU(int num_nodes, const std::string xml_filename) :
	// For now we don't allow num_nodes : override to 1 node only.
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>(1, 1.0), _count(0), _xml_filename(xml_filename) {
}

SimulationParserCPU<MPILib::CustomConnectionParameters>::SimulationParserCPU(const std::string xml_filename) :
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>(1, 1.0), _count(0), _xml_filename(xml_filename) {
}

void SimulationParserCPU<MPILib::CustomConnectionParameters>::endSimulation() {
	MPILib::MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::endSimulation();
}

void SimulationParserCPU<MPILib::CustomConnectionParameters>::addConnectionCCP(pugi::xml_node& xml_conn) {
	MPILib::CustomConnectionParameters connection;

	std::string in = std::string(xml_conn.attribute("In").value());
	std::string out = std::string(xml_conn.attribute("Out").value());

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if ((std::string("In") == std::string(ait->name())) || (std::string("Out") == std::string(ait->name())))
			continue;

		connection.setParam(std::string(ait->name()), std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.makeFirstInputOfSecond(_node_ids[in], _node_ids[out], connection);
}

void SimulationParserCPU<MPILib::CustomConnectionParameters>::addIncomingConnectionCCP(pugi::xml_node& xml_conn) {
	MPILib::CustomConnectionParameters connection;

	std::string node = std::string(xml_conn.attribute("Node").value());

	for (pugi::xml_attribute_iterator ait = xml_conn.attributes_begin(); ait != xml_conn.attributes_end(); ++ait) {

		if (std::string("Node") == std::string(ait->name()))
			continue;

		connection.setParam(std::string(ait->name()), std::string(ait->value()));
		// todo : Check the value for a variable definition - need a special function for checking all inputs really
	}

	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.setNodeExternalPrecursor(_node_ids[node], connection);
}

double SimulationParserCPU<MPILib::CustomConnectionParameters>::getCurrentSimTime() {
	return _count * MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step;
}

void SimulationParserCPU<MPILib::CustomConnectionParameters>::parseXmlFile() {
	pugi::xml_document doc;
	if (!doc.load_file(_xml_filename.c_str())) {
		std::cout << "Failed to load XML simulation file.\n";
		return; //better to throw...
	}

	//check Weight Type matches this class
	if (std::string("CustomConnectionParameters") != std::string(doc.child("Simulation").child_value("WeightType"))) {
		std::cout << "The weight type of the SimulationParser (" << "CustomConnectionParameters" << ") doesn't match the WeightType in the XML file (" << doc.child("Simulation").child_value("WeightType") << "). Exiting.\n";
		return;
	}

	//Algorithms
	//Ignore "Group" algorithms - this is the non-cuda version of WinMiind (for now)

	_algorithms = std::map<std::string, std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>>();
	_node_ids = std::map<std::string, MPILib::NodeId>();

	for (pugi::xml_node algorithm = doc.child("Simulation").child("Algorithms").child("Algorithm"); algorithm; algorithm = algorithm.next_sibling("Algorithm")) {
		//Check all possible Algorithm types
		if (std::string("GridAlgorithm") == std::string(algorithm.attribute("type").value())) {
			std::string algorithm_name = std::string(algorithm.attribute("name").value());
			std::cout << "Found GridAlgorithm " << algorithm_name << ".\n";

			std::string model_filename = std::string(algorithm.attribute("modelfile").value());
			double tau_refractive = std::stod(std::string(algorithm.attribute("tau_refractive").value()));
			std::string transform_filename = std::string(algorithm.attribute("transformfile").value());
			double start_v = std::stod(std::string(algorithm.attribute("start_v").value()));
			double start_w = std::stod(std::string(algorithm.attribute("start_w").value()));
			double time_step = std::stod(std::string(algorithm.child_value("TimeStep")));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new TwoDLib::GridAlgorithm(model_filename, transform_filename, time_step, start_v, start_w, tau_refractive));
		}

		if (std::string("MeshAlgorithmCustom") == std::string(algorithm.attribute("type").value())) {
			std::string algorithm_name = std::string(algorithm.attribute("name").value());
			std::cout << "Found MeshAlgorithmCustom " << algorithm_name << ".\n" << std::flush;

			std::string model_filename = std::string(algorithm.attribute("modelfile").value());
			double tau_refractive = std::stod(std::string(algorithm.attribute("tau_refractive").value()));
			double time_step = std::stod(std::string(algorithm.child_value("TimeStep")));

			std::vector<std::string> matrix_files;
			for (pugi::xml_node matrix_file = algorithm.child("MatrixFile"); matrix_file; matrix_file = matrix_file.next_sibling("MatrixFile")) {
				matrix_files.push_back(std::string(matrix_file.child_value()));
			}

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new TwoDLib::MeshAlgorithmCustom<TwoDLib::MasterOdeint>(model_filename, matrix_files, time_step, tau_refractive));
		}

		if (std::string("RateFunctor") == std::string(algorithm.attribute("type").value())) {
			// As we can't use the "expression" part properly here because we're not doing an intemediate cpp translation step
			// Let's just use RateAlgorithm for RateFunctor for now.
			std::string algorithm_name = std::string(algorithm.attribute("name").value());
			std::cout << "Found RateFunctor (Using a RateAlgorithm) " << algorithm_name << ".\n";

			double rate = std::stod(std::string(algorithm.child_value("expression")));

			_algorithms[algorithm_name] = std::unique_ptr<MPILib::AlgorithmInterface<MPILib::CustomConnectionParameters>>(new MPILib::RateAlgorithm<MPILib::CustomConnectionParameters>(rate));
		}

		//... todo : other algorithms
		//... todo : AvgV or no?
	}

	//Nodes
	for (pugi::xml_node node = doc.child("Simulation").child("Nodes").child("Node"); node; node = node.next_sibling("Node")) {
		std::string node_name = std::string(node.attribute("name").value());
		std::cout << "Found Node " << node_name << ".\n";

		// Check what type the node is
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

		std::string algorithm_name = std::string(node.attribute("algorithm").value());

		MPILib::NodeId id = MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.addNode(*_algorithms[algorithm_name], node_type);
		_node_ids[node_name] = id;
	}

	//Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("Connection"); conn; conn = conn.next_sibling("Connection")) {
		// A better way to do this is to move the connection building to a separate concrete non-templated class
		// too lazy right now...
		addConnectionCCP(conn);
		// todo : Deal with other connection types - DelayedConnection, double
	}

	//Incoming Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("IncomingConnection"); conn; conn = conn.next_sibling("IncomingConnection")) {
		// A better way to do this is to move the connection building to a separate concrete non-templated class
		// too lazy right now...
		addIncomingConnectionCCP(conn);
		// todo : Deal with other connection types - DelayedConnection, double
	}

	//Outgoing Connections
	for (pugi::xml_node conn = doc.child("Simulation").child("Connections").child("OutgoingConnection"); conn; conn = conn.next_sibling("OutgoingConnection")) {
		std::string node = std::string(conn.attribute("Node").value());
		MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.setNodeExternalSuccessor(_node_ids[node]);
		_ordered_output_nodes.push_back(node);
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

	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_simulation_length = simulation_length;
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step = time_step;

	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::report_handler = new MPILib::report::handler::InactiveReportHandler();

	MPILib::SimulationRunParameter par_run(*MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::report_handler, (simulation_length / time_step) + 1, 0,
		simulation_length, time_step, time_step, log_filename);

	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.configureSimulation(par_run);

}

void SimulationParserCPU<MPILib::CustomConnectionParameters>::startSimulation() {
	if (_display_nodes.size() > 0)
		TwoDLib::Display::getInstance()->animate(true, _display_nodes, MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step);
	MPILib::MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::startSimulation();
}

void SimulationParserCPU<MPILib::CustomConnectionParameters>::init() {
	parseXmlFile();
}

std::vector<double> SimulationParserCPU<MPILib::CustomConnectionParameters>::evolveSingleStep(std::vector<double> activity) {
	MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::network.reportNodeActivities(_rate_nodes, _rate_node_intervals,
		(_count * MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step));

	if (_display_nodes.size() > 0)
		TwoDLib::Display::getInstance()->updateDisplay(_count);

	TwoDLib::GridReport<MPILib::CustomConnectionParameters>::getInstance()->reportDensity(_density_nodes, _density_node_start_times, _density_node_end_times, _density_node_intervals,
		(_count * MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step));

	_count++;

	return MPILib::MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::evolveSingleStep(activity);
}

bool SimulationParserCPU<MPILib::CustomConnectionParameters>::simulationComplete() {
	return (_count * MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_time_step >=
		MiindTvbModelAbstract<MPILib::CustomConnectionParameters, MPILib::utilities::CircularDistribution>::_simulation_length);
}