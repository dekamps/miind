# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:28:18 2015

@author: scsmdk
"""
from miind.nodes import NODE_NAMES

def parse_canvas_handler(tree):
    par=tree.find('CanvasParameter')
    tmin=par.find('T_min')
    s = ''
    s += '\tconst MPILib::Time tmin = '
    s += tmin.text +';\n'

    tmax = par.find('T_max')
    s += '\tconst MPILib::Time tmax = '
    s += tmax.text +';\n'

    fmin = par.find('F_min')
    s += '\tconst MPILib::Rate fmin = '
    s += fmin.text +';\n'

    fmax = par.find('F_max')
    s += '\tconst MPILib::Rate fmax = '
    s += fmax.text +';\n'

    statemin = par.find('State_min')
    s += '\tconst MPILib::Potential statemin = '
    s += statemin.text +';\n'

    statemax = par.find('State_max')
    s += '\tconst MPILib::Potential statemax = '
    s += statemax.text +';\n'

    densemin = par.find('Dense_min')
    s += '\tconst MPILib::Potential densemin = '
    s += densemin.text +';\n'

    densemax = par.find('Dense_max')
    s += '\tconst MPILib::Potential densemax = '
    s += densemax.text +';\n'

    s += '\tMPILib::CanvasParameter par_canvas('
    s += 'tmin,'
    s += 'tmax,'
    s += 'fmin,'
    s += 'fmax,'
    s += 'statemin,'
    s += 'statemax,'
    s += 'densemin,'
    s += 'densemax);\n\n'
    return s

def add_nodes(tree):
    s= ''
    nodes=tree.findall('CanvasNode')

    for node in nodes:
        s += '\thandler.addNodeToCanvas(id_' +  str(NODE_NAMES[node.attrib['Name']]) + ');\n'
    return s

def parse_parameter(tree, outfile, enable_root, simio_tree=None):
    sim_name = tree.find('SimulationName')
    if sim_name is not None:
        name_str = sim_name.text
    else:
        name_str = "unnamed_sim"

    if(enable_root):
        s = ''
        par_check = tree.find('CanvasParameter')
        if simio_tree and par_check:
            s += parse_canvas_handler(simio_tree)
            s += '\tMPILib::report::handler::RootReportHandler handler(\"'
            s += name_str   + '\",'
            s += 'true,true,par_canvas);\n\n'
            s += add_nodes(simio_tree)
        else:
            s += '\tMPILib::report::handler::RootReportHandler handler(\"'
            s += name_str   + '\",'
            s += 'false);\n\n'
    else:
        s  = '\tMPILib::report::handler::InactiveReportHandler handler;\n'

    outfile.write(s)

    max_iter = tree.find('max_iter')
    t_begin = tree.find('t_begin')
    t_end   = tree.find('t_end')
    t_report = tree.find('t_report')
    t_step = tree.find('t_step')
    name_log = tree.find('name_log')
    t_state = tree.find('t_state_report')

    if max_iter is not None and t_begin is not None and t_report is not None and t_state is not None:

        s  = '\tSimulationRunParameter par_run( handler,'
        s += max_iter.text + ','
        s += t_begin.text + ','
        s += t_end.text + ','
        s += t_report.text + ','
        s += t_step.text + ','
        s += '\"' + name_log.text + '\",'
        s += t_state.text + ');\n'
    else:
        s  = '\tSimulationRunParameter par_run( handler,'
        s += '(unsigned int)(' + t_end.text + '/' + t_step.text + '), '
        s += '0.0, '
        s += t_end.text + ', '
        s += t_step.text + ', '
        s += t_step.text + ', '
        s += '\"' + name_log.text + '\");\n'

    outfile.write(s)
    return
