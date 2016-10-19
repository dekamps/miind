// Copyright (c) 2005 - 2015 Marc de Kamps
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
#include <iostream>
#include <cmath>
#include <limits>
#include "QuadGenerator.hpp"
#include "TransitionMatrixGenerator.hpp"

using namespace TwoDLib;

double TransitionMatrixGenerator::scale_distance = 4.0;

TransitionMatrixGenerator::TransitionMatrixGenerator
(
	const TwoDLib::MeshTree& tree,
	Uniform& uni,
	unsigned int N,
	const FidElementList& element_list
):
_tree(tree),
_uni(uni),
_N(N),
_hit_list(0),
_lost(0),
_accounted(0),
_vec_fiducial(InitializeFiducialVector(tree.MeshRef(),element_list))
{
}

void TransitionMatrixGenerator::ApplyTranslation(vector<Point>* pvec, const Point& p)
{
	for (auto it = pvec->begin(); it != pvec->end(); it++)
		(*it) += p;
}

bool TransitionMatrixGenerator::CheckHitList(const Point& p)
{
	for (auto it = _hit_list.begin(); it != _hit_list.end(); it++){
		assert(it->_cell[0] < _tree.MeshRef().NrQuadrilateralStrips());
		assert(it->_cell[1] < _tree.MeshRef().NrCellsInStrip(it->_cell[0]));
		if (_tree.MeshRef().Quad(it->_cell[0],it->_cell[1]).IsInside(p) ){
			it->_count += 1;
			return true;
		}
	}
	return false;
}


TransitionMatrixGenerator::Hit TransitionMatrixGenerator::CheckTree(const Point& pt_translated, const vector<Point>& nbs)
{
	Hit hit;
	// nbs is filled by the kdtree to expedite things. if nbs is empty,
	for (auto it = nbs.begin(); it != nbs.end(); it++){
		vector<Coordinates> vec_c = _tree.MeshRef().PointBelongsTo(*it);
		for (auto itcell = vec_c.begin(); itcell != vec_c.end(); itcell++){
			unsigned int i = (*itcell)[0];
			unsigned int j = (*itcell)[1];

			const Quadrilateral& quad = _tree.MeshRef().Quad(i,j);

			if (quad.IsInside(pt_translated)){
				hit._cell = *itcell;
				hit._count = 1;
				return hit;
			}
		}
	}
	return hit;
}

TransitionMatrixGenerator::Hit TransitionMatrixGenerator::Locate(const Point& pt_translated, const vector<Point>& nbs){
	// If the point can be located in the kdtree, or in the fiducial volume, the cell coordinates will
	// be returned with hit count 1. If linear search yields nothing, the hit count is returned as 'Lost',
	// if the point is not in a fiducial element, of 'Accounted', if it is within a fiducial volume.

	Hit hit = CheckTree(pt_translated,nbs);
	if (hit._count == 1)
		return hit;
	// else: no luck
	hit = CheckFiducial(pt_translated);
	if (hit._count == 1)
		return hit;
	// after this call, the count value may have been set to 'Accounted'.

	// Just linear search then
	for (int i = 0; i < _tree.MeshRef().NrQuadrilateralStrips(); i++)
		for (int j = 0; j < _tree.MeshRef().NrCellsInStrip(i); j++ ){
			if (_tree.MeshRef().Quad(i,j).IsInside(pt_translated) ){
				hit._count = 1;
				hit._cell = Coordinates(i,j);
				return hit;
			}
		}

	return hit;
}

void TransitionMatrixGenerator::ProcessTranslatedPoints(const vector<Point>& vec, const vector<Point>& nbs)
{
	for(auto it = vec.begin(); it != vec.end(); it++){
		if (! CheckHitList(*it) ){
			Hit h = Locate(*it,nbs);
			if (h._count > 0)
				_hit_list.push_back(h);
			else
				if (h._count == Accounted)
					_accounted.push_back(*it);
				else
					_lost.push_back(*it);
		}
	}

}

void TransitionMatrixGenerator::GenerateTransition(unsigned int strip_no, unsigned int cell_no, double v, double w)
{
	const Quadrilateral& quad = _tree.MeshRef().Quad(strip_no,cell_no);
	Point p(v,w);
	// scale_distance determines the maximum search radius
	double dist = scale_distance*DetermineDistance(quad);
	vector<Point> vec_point(_N);
	QuadGenerator gen(quad, _uni);
	gen.Generate(&vec_point);
	ApplyTranslation(&vec_point,p);
	// pick any translated point and look for points in the tree that are close, but far enough to be likely to cover the cell
	// if you want to use kdtree here, use FindNearestN here. This has been disabled because of stack size
	// problems.
	vector<Point> nbs;
	ProcessTranslatedPoints(vec_point,nbs);
}


void TransitionMatrixGenerator::Reset(unsigned int n)
{
	_N = n;
	_hit_list.clear();
}


double TransitionMatrixGenerator::DetermineDistance(const Quadrilateral& quad)
{
	double d1 = sqrt(pow(quad.Points()[0][0] - quad.Points()[2][0],2) + pow(quad.Points()[0][1] - quad.Points()[2][1],2));
	double d2 = sqrt(pow(quad.Points()[1][0] - quad.Points()[3][0],2) + pow(quad.Points()[1][1] - quad.Points()[3][1],2));

	return std::max(d1,d2);
}


vector<FiducialElement> TransitionMatrixGenerator::InitializeFiducialVector
(
	const Mesh& m,
	const FidElementList& l
) const
{
	vector<FiducialElement> vec_ret;

	for (auto it = l._vec_element.begin(); it != l._vec_element.end(); it++){
		FiducialElement el(*(it->_mesh), *(it->_quad), it->_overflow, it->_mesh->CellsBelongTo(*(it->_quad)));
		vec_ret.push_back(el);
	}
	return vec_ret;
}

bool TransitionMatrixGenerator::IsInAssociated
(
	const FiducialElement& el,
	const Point& p,
	Coordinates* pcell
)
{
	// Check if point p is in a cell associated with Fiducial element el. If yes, return true,
	// if no return false. If true is returned, the coordinates pointed to by pcell are set
	// to those of the associated cell. If false, the coordinates pointed to by pcell are set
	// to the cells with a centroid closest to point p.
	double d = std::numeric_limits<double>::max();

	for (auto itcell = el._vec_coords.begin(); itcell != el._vec_coords.end(); itcell++){
		unsigned int i = (*itcell)[0];
		unsigned int j = (*itcell)[1];
		if (el._mesh->Quad(i,j).IsInside(p)){
			(*pcell)[0] = i;
			(*pcell)[1] = j;
			return true;
		} else {
			double dist = dsquared(el._mesh->Quad(i,j).Centroid(),p);
			if (dist < d){
				d = dist;
				(*pcell)[0] = i;
				(*pcell)[1] = j;
			}
		}
	}
	return false;
}

TransitionMatrixGenerator::Hit TransitionMatrixGenerator::CheckFiducial(const Point& p){
	Hit hit;

	Coordinates closet(0,0);
	for (auto it = _vec_fiducial.begin(); it != _vec_fiducial.end(); it++){
		if (it->_quad->IsInside(p)){
			hit._count = Accounted;
			Coordinates cell;
			if (IsInAssociated(*it,p,&cell)){
				hit._count = 1;
				hit._cell =  cell;
				return hit;
			} else {
				if (it->_overflow == CONTAIN) // force it into the next closest cell
					hit._count = 1;
					hit._cell = cell;
					return hit;
			}
		}
	}

	return hit;
}

