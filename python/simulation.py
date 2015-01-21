# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:28:18 2015

@author: scsmdk
"""

def parse_simulation(tree,outfile):
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
            
    s  = '\tMPILib::report::handler::RootReportHandler handler(\"'
    s += name_str   + '\",'
    s += state_bool + ');\n'
    
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
