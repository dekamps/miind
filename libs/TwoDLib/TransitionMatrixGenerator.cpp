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
#include <algorithm>
#include "MPILib/include/TypeDefinitions.hpp"
#include "QuadGenerator.hpp"
#include "TransitionMatrixGenerator.hpp"
#include "TwoDLibException.hpp"
#include <cmath>

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

bool TransitionMatrixGenerator::CheckHitList(const Coordinates& c)
{
	// If this routine is called, it is assumed that the Point p is in the
	// Mesh, i.e. that LocatePoint returns Found on this point.

	for ( Hit& h: _hit_list){
		if( h._cell[0] == c[0] && h._cell[1] == c[1]){
			h._count += 1;
			return true;
		}
	}

	return false;
}

TransitionMatrixGenerator::SearchResult TransitionMatrixGenerator::LocatePoint(const Point& pt_translated, Coordinates* pc){
	// Walks through the mesh to see if the Point pt_tanslated can be found in the Mesh. It
	// first checks whether the point is in a Fiducial volume. If it is in one of Mesh cells associated
	// with the fiducial volume it is found; if it is within the Fiducial volume, it is lost. If it
	// is not in any Fiducial volume, the Mesh is searched, and if the cell is found, the coordinates that
	// pc point to are set to the cell corrdinates. If it is not found after an exhaustive search, it is lost,
	// and that result is retuned as Lost.
	SearchResult res = CheckFiducial(pt_translated, pc);
	if ( res == Found || res == Accounted)
		return res;

	// It wasn't in the fiducial element
	for (MPILib::Index i = 0; i < _tree.MeshRef().NrStrips(); i++)
	  for (MPILib::Index j = 0; j < _tree.MeshRef().NrCellsInStrip(i); j++ ){
			if (_tree.MeshRef().Quad(i,j).IsInside(pt_translated) ){
				*pc = Coordinates(i,j);
				return Found;
			}
		}

	return Lost;
}

void TransitionMatrixGenerator::ProcessTranslatedPoints(const vector<Point>& vec)
{
	Coordinates c(0,0);
	for(const Point& p : vec){
		SearchResult res = LocatePoint(p,&c) ;

		switch(res){
			case (Found):
				if (!CheckHitList(c)){
						Hit h;
						h._cell = c;
						h._count = 1;
						_hit_list.push_back(h);
				}
				break;
			case(Accounted):
					if (!CheckHitList(c)){
						Hit h;
						h._cell = c;
						h._count = 1;
						_hit_list.push_back(h);
				}
				_accounted.push_back(p);
				break;

			case(Lost):
				_lost.push_back(p);
				break;
			default:
				throw TwoDLibException("Unexpected result from Locate.");
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
	ProcessTranslatedPoints(vec_point);
}

void TransitionMatrixGenerator::GenerateTransitionUsingQuadTranslation(unsigned int strip_no, unsigned int cell_no, double v, double w, std::vector<Coordinates> cells)
{
	const Quadrilateral& quad = _tree.MeshRef().Quad(strip_no,cell_no);
	Point p(v,w);

	std::vector<Point> ps = quad.Points();
	Quadrilateral quad_trans = Quadrilateral((ps[0]+p), (ps[1]+p), (ps[2]+p), (ps[3]+p));

	std::vector<Point> trans_points = quad_trans.Points();
	double search_min_x = trans_points[0][0];
	double search_max_x = trans_points[0][0];
	double search_min_y = trans_points[0][1];
	double search_max_y = trans_points[0][1];

	for (Point p : trans_points) {
		if (p[0] < search_min_x)
			search_min_x = p[0];
		if (p[0] > search_max_x)
			search_max_x = p[0];
		if (p[1] < search_min_y)
			search_min_y = p[1];
		if (p[1] > search_max_y)
			search_max_y = p[1];
	}

	double total_area = 0.0;
	for (MPILib::Index c = 0; c < cells.size(); c++){
	  MPILib::Index i = cells[c][0];
		MPILib::Index j = cells[c][1];

		std::vector<Point> ps = _tree.MeshRef().Quad(i,j).Points();
		Quadrilateral quad_scaled = Quadrilateral((ps[0]), (ps[1]), (ps[2]), (ps[3]));
		std::vector<Point> ps_scaled = quad_scaled.Points();
		bool all_points_right = true;
		bool all_points_left = true;
		bool all_points_above = true;
		bool all_points_below = true;
		for(Point p : ps_scaled){
			all_points_right &= p[0] > search_max_x;
			all_points_left &= p[0] < search_min_x;
			all_points_above &= p[1] > search_max_y;
			all_points_below &= p[1] < search_min_y;
		}

		if(all_points_right || all_points_left || all_points_above || all_points_below)
			continue;

		double area = Quadrilateral::get_overlap_area(quad_trans, quad_scaled);

		if((std::abs(area))/(std::abs(quad_trans.SignedArea())) < 0.00001)
			continue;

		total_area += (std::abs(area))/(std::abs(quad_trans.SignedArea()));
		if (std::abs(area) > 0) {
			Hit h;
			h._cell = Coordinates(i,j);
			h._count = (int)(_N*(std::abs(area))/(std::abs(quad_trans.SignedArea())));
			_hit_list.push_back(h);
		}

	}

	if(_hit_list.size() == 0) {
		Hit h;
		h._cell = Coordinates(strip_no,cell_no);
		h._count = _N;
		_hit_list.push_back(h);
	}

	if (total_area != 1.0) {
		for(int i=0; i<_hit_list.size(); i++){
				_hit_list[i]._count = (int)((double)_hit_list[i]._count/total_area);
			}
	}

	// rectify rounding errors
	int c = 0;
	for(Hit h : _hit_list){
		c += h._count;
	}

	// This can create an off by one error in the count (although this is preferable
  // to mass loss) if the count is low. Eg count==100 => transition off by 0.01
	// For this reason, only use this technique
	// with large counts
	if(c != _N){
		int diff = c - _N;
		_hit_list.front()._count -= diff;
	}

}

void TransitionMatrixGenerator::GenerateTransformUsingQuadTranslation(unsigned int strip_no, unsigned int cell_no, const TwoDLib::MeshTree& trans_tree, std::vector<Coordinates> cells)
{
	const Quadrilateral& quad = trans_tree.MeshRef().Quad(strip_no,cell_no);
	Point p(0.0,0.0);

	std::vector<Point> ps = quad.Points();
	Quadrilateral quad_trans = Quadrilateral((ps[0]+p), (ps[1]+p), (ps[2]+p), (ps[3]+p));

	std::vector<Point> trans_points = quad_trans.Points();
	double search_min_x = trans_points[0][0];
	double search_max_x = trans_points[0][0];
	double search_min_y = trans_points[0][1];
	double search_max_y = trans_points[0][1];

	for (Point p : trans_points) {
		if (p[0] < search_min_x)
			search_min_x = p[0];
		if (p[0] > search_max_x)
			search_max_x = p[0];
		if (p[1] < search_min_y)
			search_min_y = p[1];
		if (p[1] > search_max_y)
			search_max_y = p[1];
	}

	double total_area = 0.0;
	for (MPILib::Index c = 0; c < cells.size(); c++){
	  MPILib::Index i = cells[c][0];
		MPILib::Index j = cells[c][1];

		std::vector<Point> ps = _tree.MeshRef().Quad(i,j).Points();
		Quadrilateral quad_scaled = Quadrilateral((ps[0]), (ps[1]), (ps[2]), (ps[3]));
		std::vector<Point> ps_scaled = quad_scaled.Points();
		bool all_points_right = true;
		bool all_points_left = true;
		bool all_points_above = true;
		bool all_points_below = true;
		for(Point p : ps_scaled){
			all_points_right &= p[0] > search_max_x;
			all_points_left &= p[0] < search_min_x;
			all_points_above &= p[1] > search_max_y;
			all_points_below &= p[1] < search_min_y;
		}

		if(all_points_right || all_points_left || all_points_above || all_points_below)
			continue;

		double area = Quadrilateral::get_overlap_area(quad_trans, quad_scaled);

		if((std::abs(area))/(std::abs(quad_trans.SignedArea())) < 0.00001)
			continue;

		total_area += (std::abs(area))/(std::abs(quad_trans.SignedArea()));
		if (std::abs(area) > 0) {
			Hit h;
			h._cell = Coordinates(i,j);
			h._count = (int)(_N*(std::abs(area))/(std::abs(quad_trans.SignedArea())));
			_hit_list.push_back(h);
		}
	}

	if(_hit_list.size() == 0) {
		Hit h;
		h._cell = Coordinates(strip_no,cell_no);
		h._count = _N;
		_hit_list.push_back(h);
	}

	if (total_area != 1.0) {
		for(int i=0; i<_hit_list.size(); i++){
				_hit_list[i]._count = (int)((double)_hit_list[i]._count/total_area);
			}
	}

	// rectify rounding errors
	int c = 0;
	for(Hit h : _hit_list){
		c += h._count;
	}

	if(c != _N){
		int diff = c - _N;
		_hit_list.front()._count -= diff;
	}

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

TransitionMatrixGenerator::SearchResult TransitionMatrixGenerator::CheckFiducial(const Point& p, Coordinates* pc){
	Hit hit;

	Coordinates closet(0,0);
	for (auto it = _vec_fiducial.begin(); it != _vec_fiducial.end(); it++){
		if (it->_quad->IsInside(p)){
			Coordinates cell;
			if (IsInAssociated(*it,p,&cell)){
				*pc =  cell;
				return Found;
			} else {
				if (it->_overflow == CONTAIN){ // force it into the next closest cell, but then it counts as found
					*pc = cell;
					return Found;
				}
				if (it->_overflow  == LEAK){    // we don't add the hit anywhere, but accept its loss and it is accounted for
					*pc=Coordinates(0,0);
					return Accounted;
				}
				throw TwoDLibException("Fiducial elements should be LEAK or CONTAIN");
			}
		}
	}

	return Lost;
}
