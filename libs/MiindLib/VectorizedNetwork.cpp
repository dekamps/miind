#include "VectorizedNetwork.hpp"
#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>
#include <chrono>

using namespace MiindLib;

VectorizedNetwork::VectorizedNetwork(MPILib::Time time_step) :
    _num_nodes(0),
    _n_steps(10),
    _network_time_step(time_step)
{
}

void VectorizedNetwork::addGridNode(TwoDLib::Mesh mesh, TwoDLib::TransitionMatrix tmat, double start_v, double start_w, double start_u,
    std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res, double tau_refractive, unsigned int finite_size) {

    _num_nodes++;
    _grid_node_ids.push_back(_num_nodes - 1);
    _grid_vec_mesh.push_back(mesh);
    _grid_vec_vec_rev.push_back(vec_rev);
    _grid_vec_vec_res.push_back(vec_res);
    _grid_vec_tau_refractive.push_back(tau_refractive);

    _vec_transforms.push_back(tmat);
    _start_vs.push_back(start_v);
    _start_ws.push_back(start_w);
    _start_us.push_back(start_u);

    _num_grid_objects.push_back(finite_size);
}

void VectorizedNetwork::addMeshNode(TwoDLib::Mesh mesh,
    std::vector<TwoDLib::Redistribution> vec_rev, std::vector<TwoDLib::Redistribution> vec_res, double tau_refractive, unsigned int finite_size) {

    _num_nodes++;
    _mesh_node_ids.push_back(_num_nodes - 1);
    _mesh_vec_mesh.push_back(mesh);
    _mesh_vec_vec_rev.push_back(vec_rev);
    _mesh_vec_vec_res.push_back(vec_res);
    _mesh_vec_tau_refractive.push_back(tau_refractive);
    _num_mesh_objects.push_back(finite_size);
}

void VectorizedNetwork::addRateNode(function_pointer functor) {
    _num_nodes++;
    _rate_functions.push_back(function_association(_num_nodes - 1, functor));
}

void VectorizedNetwork::addRateNode(rate_functor functor) {
    _num_nodes++;
    _rate_functors[_num_nodes - 1] = functor;
}

void VectorizedNetwork::initOde2DSystem(unsigned int min_solve_steps) {

    int mesh_count = 0;
    for (int g = 0; g < _grid_node_ids.size(); g++) {
        _node_id_to_group_mesh.insert(std::pair<MPILib::NodeId, MPILib::Index>(_grid_node_ids[g], mesh_count));
        _group_mesh_to_node_id.insert(std::pair<MPILib::Index, MPILib::NodeId>(mesh_count, _grid_node_ids[g]));
        _grid_meshes.push_back(mesh_count);
        mesh_count++;
        _vec_mesh.push_back(_grid_vec_mesh[g]);
        _vec_vec_rev.push_back(_grid_vec_vec_rev[g]);
        _vec_vec_res.push_back(_grid_vec_vec_res[g]);
        _vec_tau_refractive.push_back(_grid_vec_tau_refractive[g]);
        _num_objects.push_back(_num_grid_objects[g]);
    }

    for (int m = 0; m < _mesh_node_ids.size(); m++) {
        _node_id_to_group_mesh.insert(std::pair<MPILib::NodeId, MPILib::Index>(_mesh_node_ids[m], mesh_count));
        _group_mesh_to_node_id.insert(std::pair<MPILib::Index, MPILib::NodeId>(mesh_count, _mesh_node_ids[m]));
        _mesh_meshes.push_back(mesh_count);
        mesh_count++;
        _vec_mesh.push_back(_mesh_vec_mesh[m]);
        _vec_vec_rev.push_back(_mesh_vec_vec_rev[m]);
        _vec_vec_res.push_back(_mesh_vec_vec_res[m]);
        _vec_tau_refractive.push_back(_mesh_vec_tau_refractive[m]);
        _num_objects.push_back(_num_mesh_objects[m]);
    }

#ifdef IZHIKEVICH_TEST
    _network_time_step = 0.0001;
    _display_nodes.clear();
    _num_objects.clear();
    _num_objects.push_back(50000);
    _group = new TwoDLib::Ode2DSystemGroup(_vec_mesh, _vec_vec_rev, _vec_vec_res, _vec_tau_refractive, _num_objects);
#else
    _group = new TwoDLib::Ode2DSystemGroup(_vec_mesh, _vec_vec_rev, _vec_vec_res, _vec_tau_refractive, _num_objects);
#endif

    for (MPILib::Index i = 0; i < _grid_meshes.size(); i++) {
        vector<TwoDLib::Coordinates> coords = _vec_mesh[_grid_meshes[i]].findPointInMeshSlow(TwoDLib::Point(_start_vs[i], _start_ws[i]), _start_us[i]);
        _group->Initialize(_grid_meshes[i], coords[0][0], coords[0][1]);

        //create CSR Matrix for each transforms
        _csrs.push_back(TwoDLib::CSRMatrix(_vec_transforms[i], *(_group), _grid_meshes[i]));
        _grid_transform_indexes.push_back(i);
    }

    for (MPILib::Index i = 0; i < _mesh_meshes.size(); i++) {
        _group->Initialize(_mesh_meshes[i], 0, 0);
    }

    _master_steps = min_solve_steps;

    // All grids/meshes must have the same timestep
    TwoDLib::MasterParameter par(static_cast<MPILib::Number>(ceil(_network_time_step / _vec_mesh[0].TimeStep())));
    _n_steps = par._N_steps;
    std::cout << "Using master solver n_steps = " << _master_steps << "\n";

    _group_adapter = new CudaTwoDLib::CudaOde2DSystemAdapter(*(_group), _network_time_step);
}

void VectorizedNetwork::reportNodeActivities(MPILib::Time sim_time) {
    for (int i = 0; i < _rate_nodes.size(); i++) {
        if (std::fabs(std::remainder(sim_time, _rate_intervals[i])) > 0.00000001)
            continue;
        std::ostringstream ost2;
        ost2 << "rate_" << _rate_nodes[i];
        std::ofstream ofst_rate(ost2.str(), std::ofstream::app);
        ofst_rate.precision(10);
        ofst_rate << sim_time << "\t" << _current_node_rates[_rate_nodes[i]] << std::endl;
        ofst_rate.close();
    }
}

void VectorizedNetwork::reportNodeDensities(MPILib::Time sim_time) {
    for (int i = 0; i < _density_nodes.size(); i++) {
        if (sim_time < _density_start_times[i] || sim_time > _density_end_times[i] || std::fabs(std::remainder(sim_time, _density_intervals[i])) > 0.00000001)
            continue;

        std::ostringstream ost;
        ost << _density_nodes[i] << "_" << sim_time << "1";
        string fn("node_" + ost.str());

        std::string model_path("densities");
        boost::filesystem::path path(model_path);

        // MdK 27/01/2017. grid file is now created in the cwd of the program and
        // not in the directory where the mesh resides.
        const std::string dirname = path.filename().string();

        if (!boost::filesystem::exists(dirname)) {
            boost::filesystem::create_directory(dirname);
        }
        std::ofstream ofst(dirname + "/" + fn);
        _group->DumpSingleMesh(&ofst, _node_id_to_group_mesh[_density_nodes[i]]);
    }
}

void VectorizedNetwork::addGridConnection(MPILib::NodeId in, MPILib::NodeId out, std::map<std::string, std::string> params) {
    _grid_connections.push_back(NodeGridConnection(in, out, params));
}

void VectorizedNetwork::addGridConnection(MPILib::NodeId out, std::map<std::string, std::string> params, MPILib::NodeId ext_id) {
    _grid_connections.push_back(NodeGridConnection(out, params, ext_id));
}

void VectorizedNetwork::addMeshConnection(MPILib::NodeId in, MPILib::NodeId out, double efficacy, int n_conns, double delay, TwoDLib::TransitionMatrix* tmat) {
    _mesh_connections.push_back(NodeMeshConnection(in, out, efficacy, n_conns, delay, tmat));
}

void VectorizedNetwork::addMeshConnection(MPILib::NodeId out, double efficacy, int n_conns, double delay, TwoDLib::TransitionMatrix* tmat, MPILib::NodeId ext_id) {
    _mesh_connections.push_back(NodeMeshConnection(out, efficacy, n_conns, delay, tmat, ext_id));
}

void VectorizedNetwork::addMeshCustomConnection(MPILib::NodeId in, MPILib::NodeId out, std::map<std::string, std::string> params, TwoDLib::TransitionMatrix* tmat) {
    _mesh_custom_connections.push_back(NodeMeshCustomConnection(in, out, params, tmat));
}

void VectorizedNetwork::addMeshCustomConnection(MPILib::NodeId out, std::map<std::string, std::string> params, TwoDLib::TransitionMatrix* tmat, MPILib::NodeId ext_id) {
    _mesh_custom_connections.push_back(NodeMeshCustomConnection(out, params, tmat, ext_id));
}

void VectorizedNetwork::addExternalMonitor(MPILib::NodeId nid) {
    _monitored_nodes.push_back(nid);
}

void VectorizedNetwork::setupLoop(bool write_displays) {

    for (unsigned int i = 0; i < _display_nodes.size(); i++) {
        TwoDLib::Display::getInstance()->addOdeSystem(_display_nodes[i], _group, _vec_mesh[_node_id_to_group_mesh[_display_nodes[i]]].getGridNumDimensions() >= 3, _node_id_to_group_mesh[_display_nodes[i]]);
    }

    const MPILib::Time h = 1. / _master_steps * _vec_mesh[0].TimeStep();

    // Setup the OpenGL displays (if there are any required)
    if (_display_nodes.size() > 0) {
        TwoDLib::Display::getInstance()->animate(write_displays, _display_nodes, _network_time_step);
    }

    std::vector<std::vector<fptype>> grid_cell_efficacies;

    for (unsigned int i = 0; i < _grid_connections.size(); i++) {
        // for each connection, which of group's meshes is being affected
        _connection_out_group_mesh.push_back(_node_id_to_group_mesh[_grid_connections[i]._out]);
        _effs.push_back(std::stod(_grid_connections[i]._params["efficacy"]));

        int total_num_cells = 1;
        for (int d = 0; d < _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions(); d++)
            total_num_cells *= _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridResolutionByDimension(d);
        
        if (_grid_connections[i]._params.find("type") == _grid_connections[i]._params.end()) {
            std::vector<fptype> cell_vals(total_num_cells, std::stod(_grid_connections[i]._params["efficacy"]));
            grid_cell_efficacies.push_back(cell_vals);
        } else if (_grid_connections[i]._params["type"] == std::string("eff_times_v")){
            std::vector<fptype> cell_vals(total_num_cells);
            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                cell_vals[c] = std::stod(_grid_connections[i]._params["efficacy"]) *
                    ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions()-1] 
                        * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1))
                        + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1));
            }
            grid_cell_efficacies.push_back(cell_vals);
        } 
        else if (_grid_connections[i]._params["type"] == std::string("eff_times_w")) {
            std::vector<fptype> cell_vals(total_num_cells);
            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                cell_vals[c] = std::stod(_grid_connections[i]._params["efficacy"]) *
                    ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2]
                        * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2))
                        + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2));
            }
            grid_cell_efficacies.push_back(cell_vals);
        } 
        else if (_grid_connections[i]._params["type"] == std::string("eff_times_u")) {
            std::vector<fptype> cell_vals(total_num_cells);
            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                cell_vals[c] = std::stod(_grid_connections[i]._params["efficacy"]) *
                    ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3]
                        * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3))
                        + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3));
            }
            grid_cell_efficacies.push_back(cell_vals);
        }

        unsigned int connection_dimension = 0;
        if (_grid_connections[i]._params.find("dimension") != _grid_connections[i]._params.end())
            connection_dimension = std::stoi(_grid_connections[i]._params["dimension"]);

        // Calculate the offset based on dimension of the connection
        unsigned int num_dimensions = _vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions();
        unsigned int offset = 1;
        for (unsigned int i = 0; i < connection_dimension; i++)
            offset *= _vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridResolutionByDimension(num_dimensions - 1 - i);

        _grid_cell_widths.push_back(_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(num_dimensions - 1 - connection_dimension));
        _grid_cell_offsets.push_back(offset);

        _connection_queue.push_back(MPILib::DelayedConnectionQueue(_network_time_step, std::stod(_grid_connections[i]._params["delay"])));
        if (_grid_connections[i]._external)
            if (_external_to_connection_queue.find(_grid_connections[i]._external_id) == _external_to_connection_queue.end()) {
                _external_to_connection_queue.insert(
                    std::pair<MPILib::NodeId, std::vector<MPILib::NodeId>>(_grid_connections[i]._external_id, std::vector<MPILib::NodeId>()));
                _external_to_connection_queue[_grid_connections[i]._external_id].push_back(_connection_queue.size() - 1);
            }
            else {
                _external_to_connection_queue[_grid_connections[i]._external_id].push_back(_connection_queue.size() - 1);
            }
        else
            if (_node_to_connection_queue.find(_grid_connections[i]._in) == _node_to_connection_queue.end()) {
                _node_to_connection_queue.insert(
                    std::pair<MPILib::NodeId, std::vector<MPILib::NodeId>>(_grid_connections[i]._in, std::vector<MPILib::NodeId>()));
                _node_to_connection_queue[_grid_connections[i]._in].push_back(_connection_queue.size() - 1);
            }
            else {
                _node_to_connection_queue[_grid_connections[i]._in].push_back(_connection_queue.size() - 1);
            }
    }

    for (unsigned int i = 0; i < _mesh_connections.size(); i++) {
        _connection_out_group_mesh.push_back(_node_id_to_group_mesh[_mesh_connections[i]._out]);
        _csrs.push_back(TwoDLib::CSRMatrix(*(_mesh_connections[i]._transition), *(_group), _node_id_to_group_mesh[_mesh_connections[i]._out]));
        // _csrs contains all the grid transforms first (see initOde2DSystem)
        // now we're adding all the mesh transition matrices so set the correct index value
        _mesh_transform_indexes.push_back(_grid_meshes.size() + i);

        _connection_queue.push_back(MPILib::DelayedConnectionQueue(_network_time_step, _mesh_connections[i]._delay));
        if (_mesh_connections[i]._external)
            if (_external_to_connection_queue.find(_mesh_connections[i]._external_id) == _external_to_connection_queue.end()) {
                _external_to_connection_queue.insert(
                    std::pair<MPILib::NodeId, std::vector<MPILib::NodeId>>(_mesh_connections[i]._external_id, std::vector<MPILib::NodeId>()));
                _external_to_connection_queue[_mesh_connections[i]._external_id].push_back(_connection_queue.size() - 1);
            }
            else {
                _external_to_connection_queue[_mesh_connections[i]._external_id].push_back(_connection_queue.size() - 1);
            }
        else
            if (_node_to_connection_queue.find(_mesh_connections[i]._in) == _node_to_connection_queue.end()) {
                _node_to_connection_queue.insert(
                    std::pair<MPILib::NodeId, std::vector<MPILib::NodeId>>(_mesh_connections[i]._in, std::vector<MPILib::NodeId>()));
                _node_to_connection_queue[_mesh_connections[i]._in].push_back(_connection_queue.size() - 1);
            }
            else {
                _node_to_connection_queue[_mesh_connections[i]._in].push_back(_connection_queue.size() - 1);
            }
    }

    for (unsigned int i = 0; i < _mesh_custom_connections.size(); i++) {
        // for each connection, which of group's meshes is being affected
        _connection_out_group_mesh.push_back(_node_id_to_group_mesh[_mesh_custom_connections[i]._out]);

        TwoDLib::TransitionMatrix mat = *(_mesh_custom_connections[i]._transition);
        auto id = _node_id_to_group_mesh[_mesh_custom_connections[i]._out];
        auto ffs = TwoDLib::CSRMatrix(*(_mesh_custom_connections[i]._transition), *(_group), _node_id_to_group_mesh[_mesh_custom_connections[i]._out]);
        _csrs.push_back(ffs);

        // _csrs contains all the grid transforms first (see initOde2DSystem)
        // now we're adding all the mesh transition matrices so set the correct index value
        _mesh_transform_indexes.push_back(_grid_meshes.size() + i);

        _connection_queue.push_back(MPILib::DelayedConnectionQueue(_network_time_step, std::stod(_mesh_custom_connections[i]._params["delay"])));
        if (_mesh_custom_connections[i]._external)
            if (_external_to_connection_queue.find(_mesh_custom_connections[i]._external_id) == _external_to_connection_queue.end()) {
                _external_to_connection_queue.insert(
                    std::pair<MPILib::NodeId, std::vector<MPILib::NodeId>>(_mesh_custom_connections[i]._external_id, std::vector<MPILib::NodeId>()));
                _external_to_connection_queue[_mesh_custom_connections[i]._external_id].push_back(_connection_queue.size() - 1);
            }
            else {
                _external_to_connection_queue[_mesh_custom_connections[i]._external_id].push_back(_connection_queue.size() - 1);
            }
        else
            if (_node_to_connection_queue.find(_mesh_custom_connections[i]._in) == _node_to_connection_queue.end()) {
                _node_to_connection_queue.insert(
                    std::pair<MPILib::NodeId, std::vector<MPILib::NodeId>>(_mesh_custom_connections[i]._in, std::vector<MPILib::NodeId>()));
                _node_to_connection_queue[_mesh_custom_connections[i]._in].push_back(_connection_queue.size() - 1);
            }
            else {
                _node_to_connection_queue[_mesh_custom_connections[i]._in].push_back(_connection_queue.size() - 1);
            }
    }

    _csr_adapter = new CudaTwoDLib::CSRAdapter(*_group_adapter, _csrs, 
        _effs.size(), h, _mesh_transform_indexes, _grid_transform_indexes);

    _csr_adapter->InitializeStaticGridCellEfficacies(_connection_out_group_mesh, grid_cell_efficacies, _grid_cell_widths, _grid_cell_offsets);

    const auto p1 = std::chrono::system_clock::now();
    _csr_adapter->setRandomSeeds(std::chrono::duration_cast<std::chrono::seconds>(
        p1.time_since_epoch()).count());
}

std::vector<double> VectorizedNetwork::singleStep(std::vector<double> activities, unsigned int i_loop) {
    MPILib::Time time = _network_time_step * i_loop;
#ifndef IZHIKEVICH_TEST
    for (const auto& element : _rate_functions) {
        for (unsigned int i = 0; i < _node_to_connection_queue[element.first].size(); i++) {
            _connection_queue[_node_to_connection_queue[element.first][i]].updateQueue(element.second(time));
        }
    }

    for (const auto& element : _rate_functors) {
        for (unsigned int i = 0; i < _node_to_connection_queue[element.first].size(); i++) {
            _connection_queue[_node_to_connection_queue[element.first][i]].updateQueue(element.second(time));
        }
    }

    for (int i = 0; i < activities.size(); i++) {
        for (unsigned int j = 0; j < _external_to_connection_queue[i].size(); j++)
            _connection_queue[_external_to_connection_queue[i][j]].updateQueue(activities[i]);
    }

    std::vector<fptype> rates;
    int connection_count = 0;
    for (unsigned int i = 0; i < _grid_connections.size(); i++) {
        rates.push_back(_connection_queue[connection_count].getCurrentRate() * std::stod(_grid_connections[i]._params["num_connections"]));
        connection_count++;
    }
    for (unsigned int i = 0; i < _mesh_connections.size(); i++) {
        rates.push_back(_connection_queue[connection_count].getCurrentRate() * _mesh_connections[i]._n_connections);
        connection_count++;
    }

    for (unsigned int i = 0; i < _mesh_custom_connections.size(); i++) {
        rates.push_back(_connection_queue[connection_count].getCurrentRate() * std::stod(_mesh_custom_connections[i]._params["num_connections"]));
        connection_count++;
    }

    for (MPILib::Index i_part = 0; i_part < _n_steps; i_part++) {
        _group_adapter->EvolveOnDevice(_mesh_meshes);
        _group_adapter->RemapReversal();
        _group_adapter->RemapReversalFiniteObjects();

        _csr_adapter->ClearDerivative();
        _csr_adapter->SingleTransformStep();
        _csr_adapter->AddDerivativeFull();
        
        _csr_adapter->SingleTransformStepFiniteSize();
    }

    _group_adapter->RedistributeProbability(_grid_meshes);
    _group_adapter->MapFinish(_grid_meshes);

    for (unsigned int i = 0; i < rates.size(); i++)
        rates[i] *= _n_steps;

    for (MPILib::Index i_part = 0; i_part < _master_steps; i_part++) {
        _csr_adapter->ClearDerivative();
        _csr_adapter->CalculateMeshGridDerivativeWithEfficacy(_connection_out_group_mesh, rates);
        _csr_adapter->AddDerivative();
    }

    _csr_adapter->CalculateMeshGridDerivativeWithEfficacyFinite(_connection_out_group_mesh, rates, _effs, _grid_cell_widths, _grid_cell_offsets, _vec_mesh[0].TimeStep());

    _group_adapter->RedistributeFiniteObjects(_mesh_meshes, _vec_mesh[0].TimeStep(), _csr_adapter->getCurandState());
    _group_adapter->RedistributeGridFiniteObjects(_grid_meshes, _csr_adapter->getCurandState());

    _group_adapter->RedistributeProbability(_mesh_meshes);
    _group_adapter->MapFinish(_mesh_meshes);
#else
    _csr_adapter->IzhTest(_group_adapter->getSpikes());
#endif

    const std::vector<fptype>& group_rates = _group_adapter->F(_n_steps);

    for (unsigned int i = 0; i < group_rates.size(); i++) {
        _current_node_rates[_group_mesh_to_node_id[i]] = group_rates[i];
        for (unsigned int j = 0; j < _node_to_connection_queue[_group_mesh_to_node_id[i]].size(); j++) {
            _connection_queue[_node_to_connection_queue[_group_mesh_to_node_id[i]][j]].updateQueue(group_rates[i]);
        }
    }
    
    std::vector<double> monitored_rates(_monitored_nodes.size());
    for (unsigned int i = 0; i < _monitored_nodes.size(); i++) {
        monitored_rates[i] = group_rates[_node_id_to_group_mesh[_monitored_nodes[i]]];
    }
  
    if (_display_nodes.size() > 0) {
        _group_adapter->updateGroupMass();
        _group_adapter->updateFiniteObjects();
        for (MPILib::Index i_part = 0; i_part < _n_steps; i_part++) {
            _group_adapter->EvolveWithoutTransfer(_mesh_meshes);
        }
        TwoDLib::Display::getInstance()->updateDisplay(i_loop);
    }

    if (_density_nodes.size() > 0) {
        _group_adapter->updateGroupMass();
        reportNodeDensities(time);
    }

    reportNodeActivities(time);

    return monitored_rates;
}

void VectorizedNetwork::mainLoop(MPILib::Time t_begin, MPILib::Time t_end, MPILib::Time t_report, bool write_displays) {
    MPILib::Number n_iter = static_cast<MPILib::Number>(ceil((t_end - t_begin) / _network_time_step));

    setupLoop(write_displays);

    MPILib::utilities::ProgressBar* pb = new MPILib::utilities::ProgressBar(n_iter);
    boost::timer::auto_cpu_timer timer;
    for (MPILib::Index i_loop = 0; i_loop < n_iter; i_loop++) {
        singleStep(std::vector<double>(), i_loop);
        (*pb)++;
    }
}
