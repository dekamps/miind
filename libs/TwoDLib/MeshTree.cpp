// Copyright (c) 2005 - 2016 Marc de Kamps
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
#include "MeshTree.hpp"
#include "TwoDLibException.hpp"
using namespace TwoDLib;

unsigned int MeshTree::_dimension = 2;

MeshTree::MeshTree(const Mesh& mesh):
_mesh		(mesh),
_root		(0),
_visited	(0),
_distance	(0)
{
	this->FillTree();
}

vector<Point> MeshTree::FindNearestN(const Point& p, double d) const
{
	vector<Point> vec_point;

	_visited = 0;
	kd_node_t test;
	kd_node_t* found = 0;
	test.x[0] = p[0];
	test.x[1] = p[1];

	nearest(_root, &test, 0, _dimension, &found, &_distance, &_visited, d, &vec_point);

	return vec_point;
}


Point MeshTree::FindNearest(const Point& p) const
{
	_visited = 0;
	kd_node_t test;
	kd_node_t* found = 0;
	test.x[0] = p[0];
	test.x[1] = p[1];

	nearest(_root, &test, 0, _dimension, &found, &_distance, &_visited);
	return Point(found->x[0],found->x[1]);
}


void MeshTree::FillTree()
{
	vector<kd_node_t> vec_nodes;
	for(auto it_strip = _mesh._vec_vec_quad.begin(); it_strip != _mesh._vec_vec_quad.end(); it_strip++)
		for(auto it_cell = it_strip->begin(); it_cell != it_strip->end(); it_cell++){
			for(auto it_point = it_cell->Points().begin(); it_point != it_cell->Points().end(); it_point++){
				kd_node_t node;
				node.x[0] = (*it_point)[0];
				node.x[1] = (*it_point)[1];
				vec_nodes.push_back(node);
			}
		}
	_buffer = new char[vec_nodes.size()*sizeof(kd_node_t)];
	_root=make_tree(_buffer, &(vec_nodes[0]),vec_nodes.size(),0,_dimension);
}

MeshTree::~MeshTree()
{
	delete[](_buffer);
}
