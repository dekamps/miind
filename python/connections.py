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
    else:
        if weighttype.text == 'double':
            s += '\tdouble con_' + i + '_' + o + tally + '('
            s += connection.text + ');\n'
            
    s += '\tnetwork.makeFirstInputOfSecond('
    s += 'id_' + i  + ','
    s += 'id_' + o  + ','    
    s += 'con_' + i + '_' + o + tally + ');\n'
    
    return s
    
def parse_connections(connection_list,weighttype,outfile):
    for connection in connection_list:
        s = parse_connection(connection,weighttype)
        outfile.write(s)
