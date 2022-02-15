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

void VectorizedNetwork::addGridNode(TwoDLib::Mesh mesh, TwoDLib::TransitionMatrix tmat, double start_v, double start_w, double start_u, double start_x,
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
    _start_xs.push_back(start_x);

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
        vector<TwoDLib::Coordinates> coords = _vec_mesh[_grid_meshes[i]].findPointInMeshSlow(TwoDLib::Point(_start_vs[i], _start_ws[i]), _start_us[i], _start_xs[i]);
        _group->Initialize(_grid_meshes[i], coords[0][0], coords[0][1]);

        //create CSR Matrix for each transforms
        _csrs.push_back(TwoDLib::CSRMatrix(_vec_transforms[i], *(_group), _grid_meshes[i]));
        _grid_transform_indexes.push_back(i);
    }

    _vec_transforms.clear();

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

    for (int i = 0; i < _avg_nodes.size(); i++) {
        if (std::fabs(std::remainder(sim_time, _avg_intervals[i])) > 0.00000001)
            continue;

        _group_adapter->updateGroupMass();
        _group_adapter->updateFiniteObjects();
        _current_node_avgs[_node_id_to_group_mesh[_avg_nodes[i]]] = _group->Avgs(_node_id_to_group_mesh[_avg_nodes[i]]);
        
        std::ostringstream ost2;
        ost2 << "avg_" << _avg_nodes[i];
        std::ofstream ofst_rate(ost2.str(), std::ofstream::app);
        ofst_rate.precision(10);
        ofst_rate << sim_time << "\t";
        for (fptype f : _current_node_avgs[_node_id_to_group_mesh[_avg_nodes[i]]]) {
            ofst_rate << f << "\t";
        }
        ofst_rate << std::endl;
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

void VectorizedNetwork::calculateProportions1DEfficacyWithValues(double cell_width, unsigned int total_num_cells,
    std::vector<fptype>& cell_vals, int dimension_offset,
    std::vector<std::vector<fptype>>& grid_cell_efficacies, std::vector<std::vector<int>>& grid_cell_offsets,
    std::vector<inttype>& grid_cell_strides) {

    std::vector<fptype> cell_props((unsigned int)(total_num_cells * 2));
    std::vector<int> cell_offsets((unsigned int)(total_num_cells * 2));
    for (int c = 0; c < total_num_cells; c++) {

        int ofs = (int)std::abs(cell_vals[c] / cell_width);
        double goes = (double)std::fabs(cell_vals[c] / cell_width) - ofs;
        double stays = 1.0 - goes;

        int o1 = (cell_vals[c] > 0 ? ofs : -ofs) * dimension_offset;
        int o2 = (cell_vals[c] > 0 ? (ofs + 1) : (ofs - 1)) * dimension_offset;

        cell_props[modulo(c + o1, total_num_cells) * 2] = goes;
        cell_props[(modulo(c + o2, total_num_cells) * 2) + 1] = stays;
        cell_offsets[modulo(c + o1, total_num_cells) * 2] = -o1;
        cell_offsets[(modulo(c + o2, total_num_cells) * 2) + 1] = -o2;
    }

    grid_cell_efficacies.push_back(cell_props);
    grid_cell_offsets.push_back(cell_offsets);
    grid_cell_strides.push_back(2);
}

void VectorizedNetwork::generateResetRelativeNdProportions(std::vector<fptype>& final_props, std::vector<int>& final_offs, std::vector<fptype> cell_val, std::vector<double> cell_widths, std::vector<int> dimension_offset, int offset, double prop, int dim) {

    int ofs = (int)std::abs(cell_val[dim] / cell_widths[dim]);
    double goes = (double)std::fabs(cell_val[dim] / cell_widths[dim]) - ofs;
    double stays = 1.0 - goes;

    int o1 = (cell_val[dim] > 0 ? ofs : -ofs) * dimension_offset[dim];
    int o2 = (cell_val[dim] > 0 ? (ofs + 1) : -(ofs + 1)) * dimension_offset[dim];

    double n_goes = prop * goes;
    double n_stays = prop * stays;
    int n_o1 = offset + o1;
    int n_o2 = offset + o2;

    if (dim == 0) {
        if (n_goes > 0.0) {
            final_props.push_back(n_goes);
            final_offs.push_back(n_o2);
        }
        else {
            final_props.push_back(0.0);
            final_offs.push_back(0);
        }

        if (n_stays > 0.0) {
            final_props.push_back(n_stays);
            final_offs.push_back(n_o1);
        }
        else {
            final_props.push_back(0.0);
            final_offs.push_back(0);
        }
        return;
    }

    generateResetRelativeNdProportions(final_props, final_offs, cell_val, cell_widths, dimension_offset,
        n_o2, n_goes, dim - 1);

    generateResetRelativeNdProportions(final_props, final_offs, cell_val, cell_widths, dimension_offset,
        n_o1, n_stays, dim - 1);
}

TwoDLib::TransitionMatrix VectorizedNetwork::calculateProportionsNDEfficacyForCsr(TwoDLib::Mesh& mesh, std::vector<double> cell_widths, unsigned int total_num_cells,
    std::vector<std::vector<fptype>>& cell_vals, std::vector<int> dimension_offsets) {

    unsigned int num_dimensions = cell_widths.size();
    unsigned int stride = std::pow(2, num_dimensions);

    std::vector<TwoDLib::TransitionMatrix::TransferLine> lines;
    
    for (int c = 0; c < total_num_cells; c++) {
        TwoDLib::TransitionMatrix::TransferLine l;
        l._from = mesh.getStripCellCoordsOfIndex(c);

        std::vector<fptype> stride_vals;
        std::vector<int> stride_offs;

        generateResetRelativeNdProportions(stride_vals, stride_offs, cell_vals[c], cell_widths, dimension_offsets, 0, 1.0, num_dimensions - 1);

        std::map<unsigned int,  double> fracs;
        for (int k = 0; k < stride; k++) {
            unsigned int index = modulo(c + stride_offs[k], total_num_cells);
            
            if (stride_vals[k] > 0) {
                if (fracs.count(index) == 0)
                    fracs[index] = stride_vals[k];
                else
                    fracs[index] += stride_vals[k];
            }
        }

        for (auto const& x : fracs) {
            TwoDLib::TransitionMatrix::Redistribution r;
            if (mesh.cellBeyondThreshold(x.first)) {
                r._to = mesh.getStripCellCoordsOfIndex(mesh.shiftCellToThreshold(x.first));
            }
            else {
                r._to = mesh.getStripCellCoordsOfIndex(x.first);
            }  
            r._fraction = x.second;
            l._vec_to_line.push_back(r);
        }
        if (fracs.size() > 0)
            lines.push_back(l);
    }

    return TwoDLib::TransitionMatrix(lines);
}

void VectorizedNetwork::calculateProportionsNDEfficacyWithValues(std::vector<double> cell_widths, unsigned int total_num_cells,
    std::vector<std::vector<fptype>>& cell_vals, std::vector<int> dimension_offsets,
    std::vector<std::vector<fptype>>& grid_cell_efficacies, std::vector<std::vector<int>>& grid_cell_offsets,
    std::vector<inttype>& grid_cell_strides) {

    unsigned int num_dimensions = cell_widths.size();
    unsigned int stride = std::pow(2, num_dimensions);

    std::vector<fptype> cell_props((unsigned int)(total_num_cells * stride));
    std::vector<int> cell_offsets((unsigned int)(total_num_cells * stride));

    int failed_cells = 0;
    for (int c = 0; c < total_num_cells; c++) {

        std::vector<fptype> stride_vals;
        std::vector<int> stride_offs;

        generateResetRelativeNdProportions(stride_vals, stride_offs, cell_vals[c], cell_widths, dimension_offsets, 0, 1.0, num_dimensions - 1);

        for (int k = 0; k < stride; k++) {
            cell_props[modulo(c + stride_offs[k], total_num_cells) * stride + k] = stride_vals[k];
            cell_offsets[modulo(c + stride_offs[k], total_num_cells) * stride + k] = -stride_offs[k];
        }
    }

    if (failed_cells > 0)
        std::cout << "Warning: Some calculated cell transitions (" << failed_cells << ") had the same offset which may lead to mass loss.\n" ;

    grid_cell_efficacies.push_back(cell_props);
    grid_cell_offsets.push_back(cell_offsets);
    grid_cell_strides.push_back(stride);
}

void VectorizedNetwork::calculateProportionsNDEfficacyWithValuesFinite(std::vector<double> cell_widths, unsigned int total_num_cells,
    std::vector<std::vector<fptype>>& cell_vals, std::vector<int> dimension_offsets,
    std::vector<std::vector<fptype>>& grid_cell_efficacies, std::vector<std::vector<int>>& grid_cell_offsets,
    std::vector<inttype>& grid_cell_strides) {

    unsigned int num_dimensions = cell_widths.size();
    unsigned int stride = std::pow(2, num_dimensions);

    std::vector<fptype> cell_props((unsigned int)(total_num_cells * stride));
    std::vector<int> cell_offsets((unsigned int)(total_num_cells * stride));

    for (int c = 0; c < total_num_cells; c++) {

        std::vector<fptype> stride_vals;
        std::vector<int> stride_offs;

        generateResetRelativeNdProportions(stride_vals, stride_offs, cell_vals[c], cell_widths, dimension_offsets, 0, 1.0, num_dimensions - 1);

        for (int k = 0; k < stride; k++) {
            cell_props[modulo(c, total_num_cells) * stride + k] = stride_vals[k];
            cell_offsets[modulo(c, total_num_cells) * stride + k] = stride_offs[k];
        }
    }

    grid_cell_efficacies.push_back(cell_props);
    grid_cell_offsets.push_back(cell_offsets);
    grid_cell_strides.push_back(stride);
}

void VectorizedNetwork::setupLoop(bool write_displays, TwoDLib::Display * display) {

    for (unsigned int i = 0; i < _display_nodes.size(); i++) {
        display->addOdeSystem(_display_nodes[i], _group, _vec_mesh[_node_id_to_group_mesh[_display_nodes[i]]].getGridNumDimensions() >= 3, _node_id_to_group_mesh[_display_nodes[i]]);
    }

    display->setDisplayNodes(_display_nodes);

    const MPILib::Time h = 1. / _master_steps * _vec_mesh[0].TimeStep();

    std::vector<std::vector<fptype>> grid_cell_efficacies;
    std::vector<std::vector<int>> grid_cell_offsets;
    std::vector<inttype> grid_cell_strides;
    std::vector<TwoDLib::TransitionMatrix> mats;
    std::vector<TwoDLib::CSRMatrix> csrs;

    for (unsigned int i = 0; i < _grid_connections.size(); i++) {
        // for each connection, which of group's meshes is being affected
        _connection_out_group_mesh.push_back(_node_id_to_group_mesh[_grid_connections[i]._out]);
        _effs.push_back(std::stod(_grid_connections[i]._params["efficacy"]));

        int total_num_cells = 1;
        for (int d = 0; d < _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions(); d++)
            total_num_cells *= _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridResolutionByDimension(d);

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

        std::vector<double> cell_dim_widths(num_dimensions);
        for (int d = 0; d < num_dimensions; d++)
            cell_dim_widths[d] = _vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(d);

        int temp_offset = 1;
        std::vector<int> cell_dim_offs(num_dimensions);
        for (int d = num_dimensions - 1; d >= 0; d--) {
            cell_dim_offs[d] = temp_offset;
            temp_offset *= _vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridResolutionByDimension(d);
        }

        if (_grid_connections[i]._params["type"] == std::string("eff_vector_v")) {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);
            std::vector<fptype> eff_vector(num_dimensions);
            if (_grid_connections[i]._params.find("eff_v") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 1] = std::stod(_grid_connections[i]._params["eff_v"]);
            if (_grid_connections[i]._params.find("eff_w") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 2] = std::stod(_grid_connections[i]._params["eff_w"]);
            if (_grid_connections[i]._params.find("eff_u") != _grid_connections[i]._params.end() && num_dimensions>2)
                eff_vector[num_dimensions - 3] = std::stod(_grid_connections[i]._params["eff_u"]);

            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                double v = ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1]
                    * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1))
                    + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1));
                std::vector<fptype> dirs(num_dimensions);
                for (int d = 0; d < num_dimensions; d++) {
                    dirs[d] = eff_vector[d] * v;
                }

                test_cell_vals[c] = dirs;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        }
        else if (_grid_connections[i]._params["type"] == std::string("eff_vector_w")) {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);
            std::vector<fptype> eff_vector(num_dimensions);
            if (_grid_connections[i]._params.find("eff_v") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 1] = std::stod(_grid_connections[i]._params["eff_v"]);
            if (_grid_connections[i]._params.find("eff_w") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 2] = std::stod(_grid_connections[i]._params["eff_w"]);
            if (_grid_connections[i]._params.find("eff_u") != _grid_connections[i]._params.end() && num_dimensions > 2)
                eff_vector[num_dimensions - 3] = std::stod(_grid_connections[i]._params["eff_u"]);

            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                double w = ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2]
                    * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2))
                    + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2));
                std::vector<fptype> dirs(num_dimensions);
                for (int d = 0; d < num_dimensions; d++) {
                    dirs[d] = eff_vector[d] * w;
                }

                test_cell_vals[c] = dirs;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
            
        }
        else if (_grid_connections[i]._params["type"] == std::string("eff_vector_u")) {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);
            std::vector<fptype> eff_vector(num_dimensions);
            if (_grid_connections[i]._params.find("eff_v") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 1] = std::stod(_grid_connections[i]._params["eff_v"]);
            if (_grid_connections[i]._params.find("eff_w") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 2] = std::stod(_grid_connections[i]._params["eff_w"]);
            if (_grid_connections[i]._params.find("eff_u") != _grid_connections[i]._params.end() && num_dimensions > 2)
                eff_vector[num_dimensions - 3] = std::stod(_grid_connections[i]._params["eff_u"]);

            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                double u = ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3]
                    * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3))
                    + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3));
                std::vector<fptype> dirs(num_dimensions);
                for (int d = 0; d < num_dimensions; d++) {
                    dirs[d] = eff_vector[d] * u;
                }
                
                test_cell_vals[c] = dirs;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));     
        }
        else if (_grid_connections[i]._params["type"] == std::string("eff_vector")) {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);
            std::vector<fptype> eff_vector(num_dimensions);
            if (_grid_connections[i]._params.find("eff_v") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 1] = std::stod(_grid_connections[i]._params["eff_v"]);
            if (_grid_connections[i]._params.find("eff_w") != _grid_connections[i]._params.end())
                eff_vector[num_dimensions - 2] = std::stod(_grid_connections[i]._params["eff_w"]);
            if (_grid_connections[i]._params.find("eff_u") != _grid_connections[i]._params.end() && num_dimensions > 2)
                eff_vector[num_dimensions - 3] = std::stod(_grid_connections[i]._params["eff_u"]);

            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                test_cell_vals[c] = eff_vector;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        }
        else if (_grid_connections[i]._params["type"] == std::string("eff_times_v")){
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);
            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                std::vector<fptype> eff_vector(num_dimensions,0.0);

                double e = std::stod(_grid_connections[i]._params["efficacy"]) *
                    ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions()-1] 
                        * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1))
                        + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 1));

                eff_vector[num_dimensions - 1] = e;
                test_cell_vals[c] = eff_vector;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        } 
        else if (_grid_connections[i]._params["type"] == std::string("eff_times_w")) {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);

            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                std::vector<fptype> eff_vector(num_dimensions, 0.0);

                double e = std::stod(_grid_connections[i]._params["efficacy"]) *
                    ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2]
                        * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2))
                        + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 2));

                eff_vector[num_dimensions - 2] = e;
                test_cell_vals[c] = eff_vector;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        } 
        else if (_grid_connections[i]._params["type"] == std::string("eff_times_u")) {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);

            double cell_width = _grid_cell_widths.back();
            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                std::vector<fptype> eff_vector(num_dimensions, 0.0);

                double e = std::stod(_grid_connections[i]._params["efficacy"]) *
                    ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3]
                        * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3))
                        + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridNumDimensions() - 3));

                eff_vector[num_dimensions - 3] = e;
                test_cell_vals[c] = eff_vector;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        }
        else if (_grid_connections[i]._params["type"] == std::string("tsodyks")) { //specific connection type designed for the tsodyks markram 4D model
        
        std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);

        double U_se = 0.55; //55
        double A_se = 5.3; //530

        if (_grid_connections[i]._params.find("U_se") != _grid_connections[i]._params.end())
            U_se = std::stod(_grid_connections[i]._params["U_se"]);

        if (_grid_connections[i]._params.find("A_se") != _grid_connections[i]._params.end())
            A_se = std::stod(_grid_connections[i]._params["A_se"]);

        for (int c = 0; c < total_num_cells; c++) {

            std::vector<fptype> eff_vector(num_dimensions);

            double R = ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[0]
                * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(0))
                + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(0));

            double E = ((_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getCoordsOfIndex(c)[1]
                * _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridCellWidthByDimension(1))
                + _grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]].getGridBaseByDimension(1));

            eff_vector[0] = -U_se * R;
            eff_vector[1] =  U_se * R;
            eff_vector[2] =  U_se * A_se * E;
            eff_vector[3] =  0.0;

            test_cell_vals[c] = eff_vector;
        }

        mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
            cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

        csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        }
        else {
            std::vector<std::vector<fptype>> test_cell_vals(total_num_cells);
            // Calculate the v for each cell in the grid
            for (int c = 0; c < total_num_cells; c++) {
                std::vector<fptype> eff_vector(num_dimensions, 0.0);

                double e = std::stod(_grid_connections[i]._params["efficacy"]);

                eff_vector[num_dimensions - 1 - connection_dimension] = e;
                test_cell_vals[c] = eff_vector;
            }

            mats.push_back(calculateProportionsNDEfficacyForCsr(_grid_vec_mesh[_node_id_to_group_mesh[_grid_connections[i]._out]],
                cell_dim_widths, total_num_cells, test_cell_vals, cell_dim_offs));

            csrs.push_back(TwoDLib::CSRMatrix(mats.back(), *_group, _node_id_to_group_mesh[_grid_connections[i]._out]));
        }

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

    _csr_adapter->InitializeStaticGridCellCsrNd(_connection_out_group_mesh, csrs);

    csrs.clear();
    _csrs.clear();

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

    for (unsigned int i = 0; i < rates.size(); i++)
        rates[i] *= _n_steps;

    for (MPILib::Index i_part = 0; i_part < _master_steps; i_part++) {
        _csr_adapter->ClearDerivative();
        _csr_adapter->CalculateMeshGridDerivativeWithEfficacy(_connection_out_group_mesh, rates);
        _csr_adapter->AddDerivative();
    }

    _csr_adapter->CalculateMeshGridDerivativeWithEfficacyFinite(_connection_out_group_mesh, rates, _effs, _grid_cell_widths, _grid_cell_offsets, _vec_mesh[0].TimeStep());

    _group_adapter->RedistributeFiniteObjects(_mesh_meshes, _vec_mesh[0].TimeStep(), _n_steps, _csr_adapter->getCurandState());
    _group_adapter->RedistributeGridFiniteObjects(_grid_meshes, _n_steps, _csr_adapter->getCurandState());

    _group_adapter->RedistributeProbability(_mesh_meshes);
    _group_adapter->MapFinish(_mesh_meshes);

    _group_adapter->RedistributeProbability(_grid_meshes);
    _group_adapter->MapFinish(_grid_meshes);
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

    setupLoop(write_displays, TwoDLib::Display::getInstance());

    MPILib::utilities::ProgressBar* pb = new MPILib::utilities::ProgressBar(n_iter);
    boost::timer::auto_cpu_timer timer;
    for (MPILib::Index i_loop = 0; i_loop < n_iter; i_loop++) {
        singleStep(std::vector<double>(), i_loop);
        (*pb)++;
    }
}

void VectorizedNetwork::endSimulation() {
    _vec_transforms.clear();
    _start_vs.clear();
    _start_ws.clear();
    _start_us.clear();
    _start_xs.clear();

    _vec_mesh.clear();
    _vec_vec_rev.clear();
    _vec_vec_res.clear();
    _vec_tau_refractive.clear();

    _grid_node_ids.clear();
    _grid_vec_mesh.clear();
    _grid_vec_vec_rev.clear();
    _grid_vec_vec_res.clear();
    _grid_vec_tau_refractive.clear();

    _mesh_node_ids.clear();
    _mesh_vec_mesh.clear();
    _mesh_vec_vec_rev.clear();
    _mesh_vec_vec_res.clear();
    _mesh_vec_tau_refractive.clear();

    _rate_func_node_ids.clear();

    _grid_transform_indexes.clear();
    _mesh_transform_indexes.clear();

    _grid_meshes.clear();
    _mesh_meshes.clear();

    _num_grid_objects.clear();
    _num_mesh_objects.clear();
    _num_objects.clear();


    delete _group_adapter;
    delete _csr_adapter;

    _csrs.clear();

    _display_nodes.clear();
    _rate_nodes.clear();
    _avg_nodes.clear();
    _rate_intervals.clear();
    _avg_intervals.clear();
    _density_nodes.clear();
    _density_start_times.clear();
    _density_end_times.clear();
    _density_intervals.clear();

    _connection_out_group_mesh.clear();
    _connection_queue.clear();
    _node_to_connection_queue.clear();
    _external_to_connection_queue.clear();
    _effs.clear();
    _grid_cell_widths.clear();
    _grid_cell_offsets.clear();

    _current_node_rates.clear();
    _current_node_avgs.clear();

    _monitored_nodes.clear();

    _mesh_connections.clear();
    _mesh_custom_connections.clear();
    _grid_connections.clear();

    _rate_functions.clear();
    _rate_functors.clear();

    _node_id_to_group_mesh.clear();
    _group_mesh_to_node_id.clear();
}
