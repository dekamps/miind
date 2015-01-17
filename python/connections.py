# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 16:51:56 2015

@author: scsmdk
"""

import nodes

def parse_connection(connection):
    i = str(nodes.NODE_NAMES[connection.attrib['In']])
    o = str(nodes.NODE_NAMES[connection.attrib['Out']])    

    s = '\tDelayedConnection con_' + i + '_' + o +'('
    s += connection.text.split()[0] + ','
    s += connection.text.split()[1] + ','
    s += connection.text.split()[1] + ');\n'    
    s += '\tnetwork.makeFirstInputOfSecond('
    s += 'id_' + i  + ','
    s += 'id_' + o  + ','    
    s += 'con_' + i + '_' + o + ');\n'
    
    return s
    
def parse_connections(connection_list,weighttype,outfile):
    for connection in connection_list:
        s = parse_connection(connection)
        outfile.write(s)