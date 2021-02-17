# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 12:24:06 2015

@author: scsmdk
"""
import miind.algorithms as algorithms

NODE_NAMES = {}

def parse_node(node,weighttype,i):
    if node.attrib['algorithm'] not in algorithms.ALGORITHM_NAMES:
        raise Exception('No algorithm named \'' + node.attrib['algorithm'] + '\' for node \'' + node.attrib['name'] + '\'')

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

def reset_nodes():
    NODE_NAMES.clear()
