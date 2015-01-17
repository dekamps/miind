# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 12:24:06 2015

@author: scsmdk
"""
import algorithms

NODE_NAMES = {}

def parse_node(node,weighttype,i):
    s =''
    s += '\tMPILib::NodeId id_' + str(i) + ' = '
    s += 'network.addNode('
    s += algorithms.ALGORITHM_NAMES[node.attrib['algorithm']] + ','
    s += 'MPILib::'
    s += node.attrib['type'] + ');\n'
    
    NODE_NAMES[node.attrib['name']] = i
    return s
    
def parse_nodes(node_list, weighttype, outfile):
    s = ''
    i = 0
    for node in node_list:
        s = parse_node(node,weighttype,i)
        i+=1
        outfile.write(s)
