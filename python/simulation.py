# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:28:18 2015

@author: scsmdk
"""
from nodes import NODE_NAMES

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

def parse_simulation(tree,outfile,enable_root):
    name=tree.find('SimulationName')
    name_str = name.text
    state=tree.find('WithState')

    state_bool = ''
    if state.text == 'TRUE':
        state_bool='true'
    else:
        if state.text == 'FALSE':
            state_bool='false'
        else:
            raise NameError('Cannot interpret WithState')

    screen=tree.find('OnScreen')
    if screen.text == 'TRUE':
        s = parse_canvas_handler(tree)
        if(enable_root):
            s += '\tMPILib::report::handler::RootReportHandler handler(\"'
            s += name_str + '\",'
            s += state_bool + ','
            s += 'true, par_canvas);\n'
            s += add_nodes(tree)
        else:
            s += '\tMPILib::report::handler::InactiveReportHandler handler;\n'
    else:
        if(enable_root):
            s  = '\tMPILib::report::handler::RootReportHandler handler(\"'
            s += name_str   + '\",'
            s += state_bool + ');\n\n'
        else:
            s  = '\tMPILib::report::handler::InactiveReportHandler handler;\n'

    outfile.write(s)
    return

def parse_parameter(tree,outfile):
    max_iter = tree.find('max_iter')

    s  = '\tSimulationRunParameter par_run( handler,'
    s += max_iter.text + ','

    t_begin = tree.find('t_begin')
    s += t_begin.text + ','

    t_end   = tree.find('t_end')
    s += t_end.text + ','

    t_report = tree.find('t_report')
    s += t_report.text + ','

    t_step = tree.find('t_step')
    s += t_step.text + ','

    name_log = tree.find('name_log')
    s += '\"' + name_log.text + '\",'

    t_state = tree.find('t_state_report')
    s += t_state.text + ');\n'

    outfile.write(s)
    return
