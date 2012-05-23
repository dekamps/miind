// Copyright (c) 2005 - 2011 Marc de Kamps
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation 
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software 
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY 
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#ifndef _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_INCLUDE_GUARD
#define _CODE_LIBS_SPARSEIMPLEMENTATIONLIB_INCLUDE_GUARD

#include "AbstractSparseNodeCode.h"
#include "LayeredSparseImplementationCode.h"
#include "ReversibleLayeredSparseImplementationCode.h"
#include "SparseImplementationAllocator.h"
#include "SparseImplementationCode.h"
#include "SparseNodeCode.h"
#include "SparseLibException.h"
#include "SparseLibTestOpenException.h"
#include "SparsePredecessorIteratorCode.h"


/*! \page SparseImplementationLib SparseImplementationLib
 * 
 * \section sparseimp_lib_desc Brief description
 *  SparseImplementationLib's purpose is the efficient representation of sparse networks, which is crucial for any biological model,
 *  so also for neuronal networks. It creates facilities for creating networks, reading them from and writing them to disk and copying them.
 *  Although these are relatively mundane tasks, this library can be quite useful in low level code. It is also the basis for many of the
 *  other libraries. The key concept is the AbstractSparseNodes, which maintains a list of pointer-weight pairs. Each pointer relates to
 *  another AbstractSparseNode, its corresponding weight expresses the strength of the directed link. A collection of AbstractSparseNodes,
 *  a SparseImplementation represents a sparse network, if the pointers are used consistently to relate to other nodes in this collection.
 *  SparseImplementationLib checks this consistency. The persistency of the pointer-weight pairs is handled correctly by SparseImplementationLib.
*
* \section intro_sec Introduction
*
* Sparse networks occur in nearly every branch of science, be it physics, biology, economics, etc. In all
* of these areas, processes are modelled which are essentially network processes: any coupled system of
* equations is essentially a network process. And since connections take up resources (cables, white matter,
* bandwith), large networks tend to be sparse. 
*
* In creating the simulation, the emphasis is usually at the difficult bit, the things that happen at the
* nodes. Later on, if this works correctly, nodes are connected with each other and one can study the network
* process. However, if the networks become large, one usually discovers that memory becomes an issues and that
* specialized sparse network representations are required. At this stage of the processes, this can be a bit
* of a disappointment, because it takes away attention from interesting stuff to mundane programming.
* As one progresses, it is not unusual to discover that the network representations are tricky, difficult
* to program and to maintain and quite error prone. After having done this for one type of network, it is not unusual
* to find that one has to go through many familiar motions in programming other networks and the question which
* then occurs is, 'do I really have to do all this from scratch again ?'. The answer is: 'NO'.
*
* Modern object oriented languages, like C++ offer facilties to treat a network as a generic object and generic
* code that serves for the representation of sparse networks, serialization (reading from and writing to disk) and
* visualisation can be written. This is what SparseImplementationLib is about: a generic representation of sparse
* networks that serves neural networks just as well as genetic, telecommunication, social or economic networks.
* It doesn't concern itself with simulating whatever process takes place at the nodes. It is possible to use the
* higher level library DynamicLib for that. SparseImplementationLib provides very simple lightweight for network 
* representation and therefore it is relatively simple to use.
*
*
* \section intro_fac The sparse network memory model
* 
* Any network is a collection of nodes and edges, connecting the nodes. Typically, there are numerical values
* associated with the edges, and a typical computer implementation consists of a vector for storing the values
* of the nodes and a so-called adjecency matrix (or weight matrix) for storing the values of the edges (or weights). 
* This is a very general representation
* for networks, which allows fast access to both the edge values and the node values, and is easy
* to program. This representation is quite wasteful, however, when the network is sparse, i.e. when most of the
* entries in the weight matrix are zero. Biological networks are typically large and sparse (since the connections
* must be physically implemented) and realistic simulations, relying on such a network representation would quickly
* run out of memory.
*
* Biological networks are usually irregular. Symmetries, such as translation or rotation invariance which could lead
* to clever representations in artificial networks, usually are only realised approximately in biological networks
* and there is no other option than to really represent each connection between two nodes. But non-existing connections 
* need not be represented! The key idea for an efficient representation of an irregular sparse network is illustrated
* in the figure below.
*
* \image html SparseNode.png
*
*
* Each node becomes the responsibility for representing its numerical value (or activation). On top of that, the node
* maintains a list of connections. Each connection is a pointer-value pair. The pointer is a reference (not a C++ reference!)
* to a node which is connected to this node, its predecessor; the value is the numerical value of the edge (weight) 
* connecting the predecessor node to this one. So a node keeps a list of each predecessor nodes (and weights) and is thereby 
* responsible for maintaining which nodes are connected to it. It is clear that a collection of such nodes form an 
* implicit network representation. It is an implicit network representation because there is no information about the network 
* associated with each node; information on a network must be obtained by visiting each node. An explicit network representation
* entails the grouping of nodes that make up a network in a common class. First of all, this makes it possible
* to query the network at its properties. Second, it disambiguates the concept of a network (a single node might belong
* to two networks, in principle). Also, it makes it possible the notion of consistency when defining  a network (if a 
* node contains a reference to a predecessor node that is not in the network, the network is not consistent).
*
* \section intro_cpp A C++ implementation
* It is clear that a node is a good candidate for a class. In fact the class is called AbstractSparseNode. It has methods
* for setting and retrieving it activation value. Also connections can be added to the node. Example code is shown
* below. It is recommended that you don't code like this! The way to create networks is to use a SparseImplementation,
* which shields you from the naked pointers; it offers a save copy operation for networks, for example.
*/  



#endif // include guard
