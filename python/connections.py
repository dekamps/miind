# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 16:51:56 2015

@author: scsmdk
"""

import nodes

TALLY = {}

def register(i,o):
    tup = (i,o)
    if tup not in TALLY:
        TALLY[tup] = 1
        return 0
    else:
        TALLY[tup] += 1
        return TALLY[tup] - 1

def parse_connection(connection, weighttype):

    i = str(nodes.NODE_NAMES[connection.attrib['In']])
    o = str(nodes.NODE_NAMES[connection.attrib['Out']])

    # Multiple connections with same label are allowed, so we need to keep a tally
    count = register(i,o)
    tally = '_' + str(count)

    s = ''
    if weighttype.text == 'DelayedConnection':
        s += '\tDelayedConnection con_' + i + '_' + o + tally + '('
        s += connection.text.split()[0] + ','
        s += connection.text.split()[1] + ','
        s += connection.text.split()[2] + ');\n'
    elif weighttype.text == "CustomConnectionParameters":
        s += '\tCustomConnectionParameters con_' + i + '_' + o + tally + ';\n'
        for ak,av in connection.attrib.items():
            if ak == 'In' or ak == 'Out':
                continue
            s += '\tcon_' + i + '_' + o + tally + '.setParam(\"' + ak + '\", \"' + av +'\");\n'
    else:
        if weighttype.text == 'double':
            s += '\tdouble con_' + i + '_' + o + tally + '('
            s += connection.text + ');\n'

    s += '\tnetwork.makeFirstInputOfSecond('
    s += 'id_' + i  + ','
    s += 'id_' + o  + ','
    s += 'con_' + i + '_' + o + tally + ');\n'

    return s

def parse_external_outgoing_connection(connection, nodemap, network_name='network'):
    o = str(nodemap[connection.attrib['Node']])

    return '\t' + network_name + '.addExternalMonitor('+ o +');\n'


def parse_grid_connection(connection, nodemap, network_name='network'):
    i = str(nodemap[connection.attrib['In']])
    o = str(nodemap[connection.attrib['Out']])
    eff = connection.attrib['efficacy']
    num_cons = connection.attrib['num_connections']
    delay = connection.attrib['delay']

    return '\t' + network_name + '.addGridConnection('+ i +','+ o +','+ eff +','+ num_cons +','+ delay +');\n'

def parse_external_incoming_grid_connection(connection, nodemap, id, network_name='network'):
    o = str(nodemap[connection.attrib['Node']])

    eff = connection.attrib['efficacy']
    num_cons = connection.attrib['num_connections']
    delay = connection.attrib['delay']

    return '\t' + network_name + '.addGridConnection('+ o +','+ eff +','+ num_cons +',(double)'+ delay +','+ str(id) +');\n'

def parse_grid_vectorized_connection(connection, nodemap, network_name='network'):
    i = str(nodemap[connection.attrib['In']])
    o = str(nodemap[connection.attrib['Out']])
    s = '\tstd::map<std::string, std::string> params_' + i  + '_' + o + ';\n'
    for ak,av in connection.attrib.items():
        if ak in ['In', 'Out']:
            continue
        s += '\tparams_' + i  + '_' + o + '[\"' + ak + '\"] = \"' + av + '\";\n'

    s += '\t' + network_name + '.addGridConnection('+ i +','+ o +', params_' + i  + '_' + o + ');\n'
    return s

def parse_external_incoming_grid_vectorized_connection(connection, nodemap, id, network_name='network'):
    o = str(nodemap[connection.attrib['Node']])

    s = '\tstd::map<std::string, std::string> params_extern_' + o + ';\n'
    for ak,av in connection.attrib.items():
        if ak in ['Node']:
            continue
        s += '\tparams_extern_' + o + '[\"' + ak + '\"] = \"' + av + '\";\n'

    s += '\t' + network_name + '.addGridConnection('+ i +', params_extern_' + o + ',' + str(id) + ');\n'
    return s

def parse_mesh_connection(connection, nodemap, mat_name, network_name='network'):
    i = str(nodemap[connection.attrib['In']])
    o = str(nodemap[connection.attrib['Out']])
    num_cons = connection.text.split()[0]
    eff = connection.text.split()[1]
    delay = connection.text.split()[2]

    return '\t' + network_name + '.addMeshConnection('+ i +','+ o +','+ eff +','+ num_cons +','+delay+',&'+ mat_name +');\n'

def parse_external_incoming_mesh_connection(connection, nodemap, mat_name, id, network_name='network'):
    o = str(nodemap[connection.attrib['Node']])
    num_cons = connection.text.split()[0]
    eff = connection.text.split()[1]
    delay = connection.text.split()[2]

    return '\t' + network_name + '.addMeshConnection('+ o +','+ eff +','+ num_cons +',(double)'+delay+',&'+ mat_name +','+ str(id) +');\n'

def parse_mesh_vectorized_connection(connection, nodemap, mat_name, network_name='network'):
    i = str(nodemap[connection.attrib['In']])
    o = str(nodemap[connection.attrib['Out']])
    s = '\tstd::map<std::string, std::string> params_' + i  + '_' + o + ';\n'
    for ak,av in connection.attrib.items():
        if ak in ['In', 'Out']:
            continue
        s += '\tparams_' + i  + '_' + o + '[\"' + ak + '\"] = \"' + av + '\";\n'

    s += '\t' + network_name + '.addGridConnection('+ i +','+ o +', params_' + i  + '_' + o + ',&'+ mat_name +');\n'
    return s

def parse_external_incoming_mesh_vectorized_connection(connection, nodemap, mat_name, id, network_name='network'):
    o = str(nodemap[connection.attrib['Node']])

    s = '\tstd::map<std::string, std::string> params_extern_' + o + ';\n'
    for ak,av in connection.attrib.items():
        if ak in ['Node']:
            continue
        s += '\tparams_extern_' + o + '[\"' + ak + '\"] = \"' + av + '\";\n'

    s += '\t' + network_name + '.addGridConnection('+ i +', params_extern_' + o + ',&'+ mat_name +',' + str(id) + ');\n'
    return s

def parse_connections(connection_list,weighttype,outfile):
    for connection in connection_list:
        s = parse_connection(connection,weighttype)
        outfile.write(s)

def parse_incoming_connections(connection_list,weighttype,outfile):
    for connection in connection_list:
        s = parse_incoming_connection(connection,weighttype)
        outfile.write(s)

def parse_outgoing_connections(connection_list,outfile):
    for connection in connection_list:
        s = parse_outgoing_connection(connection)
        outfile.write(s)

def parse_incoming_connection(connection, weighttype):

    node = str(nodes.NODE_NAMES[connection.attrib['Node']])

    # Multiple connections with same label are allowed, so we need to keep a tally
    count = register('EXTERNAL',node)
    tally = '_' + str(count)

    s = ''
    if weighttype.text == 'DelayedConnection':
        s += '\tDelayedConnection con_EXTERNAL_' + node + tally + '('
        s += connection.text.split()[0] + ','
        s += connection.text.split()[1] + ','
        s += connection.text.split()[2] + ');\n'
    elif weighttype.text == "CustomConnectionParameters":
        s += '\tCustomConnectionParameters con_EXTERNAL_' + node + tally + ';\n'
        for ak,av in connection.attrib.items():
            if ak == 'Node':
                continue
            s += '\tcon_EXTERNAL_' + node + tally + '.setParam(\"' + ak + '\", \"' + av +'\");\n'
    else:
        if weighttype.text == 'double':
            s += '\tdouble con_EXTERNAL_' + node + tally + '('
            s += connection.text + ');\n'

    s += '\tnetwork.setNodeExternalPrecursor('
    s += 'id_' + node + ','
    s += 'con_EXTERNAL_' + node + tally + ');\n'

    return s

def parse_outgoing_connection(connection):

    node = str(nodes.NODE_NAMES[connection.attrib['Node']])

    # Multiple connections with same label are allowed, so we need to keep a tally
    count = register(node,'EXTERNAL')
    tally = '_' + str(count)

    s = ''
    s += '\tnetwork.setNodeExternalSuccessor('
    s += 'id_' + node + ');\n'

    return s
