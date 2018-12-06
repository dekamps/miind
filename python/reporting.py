def define_display_nodes(tree):
    s  = '\tstd::vector<MPILib::NodeId> display_nodes;\n'

    display_nodes = tree.find('Reporting/Display')
    for dn in display_nodes:
        node_id = nodes.NODE_NAMES[dn.attrib['node']]
        s += '\tdisplay_nodes.push_back('+ node_id + ');\n'

    s += '\n'
    return s

def define_rate_nodes(tree):
    s  = '\tstd::vector<MPILib::NodeId> rate_nodes;\n'

    rate_nodes = tree.find('Reporting/Rate')
    for rn in rate_nodes:
        node_id = nodes.NODE_NAMES[rn.attrib['node']]
        s += '\trate_nodes.push_back('+ node_id + ');\n'

    s += '\n'
    return s

def define_density_nodes(tree):
    s  = '\tstd::vector<MPILib::NodeId> density_nodes;\n'
    s  = '\tstd::vector<MPILib::Time> density_node_start_times;\n'
    s  = '\tstd::vector<MPILib::Time> density_node_end_times;\n'
    s  = '\tstd::vector<MPILib::Time> density_node_intervals;\n'

    density_nodes = tree.find('Reporting/Density')
    for dn in density_nodes:
        node_id = nodes.NODE_NAMES[dn.attrib['node']]
        t_start = nodes.NODE_NAMES[dn.attrib['t_start']]
        t_end = nodes.NODE_NAMES[dn.attrib['t_end']]
        t_interval = nodes.NODE_NAMES[dn.attrib['t_interval']]
        s += '\tdensity_nodes.push_back('+ node_id + ');\n'
        s += '\tdensity_node_start_times.push_back('+ t_start + ');\n'
        s += '\tdensity_node_end_times.push_back('+ t_end + ');\n'
        s += '\tdensity_node_intervals.push_back('+ t_interval + ');\n'

    s += '\n'
    return s
