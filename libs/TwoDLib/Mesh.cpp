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

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <cassert>
#include <limits>
#include <boost/lexical_cast.hpp>
#include "pugixml.hpp"
#include "Mesh.hpp"
#include "Quadrilateral.hpp"
#include "TwoDLibException.hpp"

using namespace TwoDLib;
using namespace std;

const int Mesh::_dimension = 2;

Mesh::Mesh(double timestep,
	std::vector<unsigned int> resolution,
	std::vector<double> dimension,
	std::vector<double> base) :
	_t_step(timestep),
	_resolution(resolution),
	_dimensions(dimension),
	_base(base){

	_grid_num_dimensions = _dimensions.size();
	_num_strips = 1;
	for (unsigned int d = 0; d < _resolution.size()-1; d++) { _num_strips *= _resolution[d]; }

	_strip_length = _resolution[_grid_num_dimensions - 1];

	for (unsigned int i = 0; i < _num_strips; i++) {
		std::vector<Cell> strip;
		for (unsigned int j = 0; j < _strip_length; j++) {
			// For now, just generate four 2D points for the last two
			// demensions
			double v_width = _dimensions[_grid_num_dimensions - 1] / _resolution[_grid_num_dimensions - 1];
			double w_width = _dimensions[0] / (_resolution[0]);
			double pv = v_width * j;
			double pw = w_width * i;
			double bv = _base[_grid_num_dimensions - 1];
			double bw = _base[0];

			std::vector<Point> ps;
			ps.push_back(Point(bv + pv, bw + pw));
			ps.push_back(Point(bv + pv + v_width, bw + pw));
			ps.push_back(Point(bv + pv + v_width, bw + pw + w_width));
			ps.push_back(Point(bv + pv, bw + pw + w_width));

			strip.push_back(Cell(ps));
		}
		_vec_vec_quad.push_back(strip);
	}
}

Mesh::Mesh(istream& s):
_vec_block(0),
_vec_vec_quad(0),
_vec_vec_gen(0),
_vec_timefactor(0){
	this->FromXML(s);
}

Mesh::Mesh(const Mesh& m):
_vec_block(m._vec_block),
_vec_vec_quad(m._vec_vec_quad),
_vec_vec_gen(m._vec_vec_gen),
_vec_timefactor(m._vec_timefactor),
_t_step(m._t_step),
_map(m._map),
_vec_vec_cell(m._vec_vec_cell),
_grid_num_dimensions(m._grid_num_dimensions),
_resolution(m._resolution),
_dimensions(m._dimensions),
_threshold_reset_dimension(m._threshold_reset_dimension),
_threshold_reset_jump_dimension(m._threshold_reset_jump_dimension),
_base(m._base),
_num_strips(m._num_strips),
_strip_length(m._strip_length),
_strips_are_v_oriented(m._strips_are_v_oriented),
_has_defined_strips(m._has_defined_strips)
{
}

double Mesh::getGridCellWidthByDimension(unsigned int dim) const {
	return _dimensions[dim] / _resolution[dim];
}

unsigned int Mesh::getGridResolutionByDimension(unsigned int dim) const {
	return _resolution[dim];
}

double Mesh::getGridBaseByDimension(unsigned int dim) const {
	return _base[dim];
}

double Mesh::getGridSizeByDimension(unsigned int dim) const {
	return _dimensions[dim];
}

unsigned int Mesh::getGridThresholdResetDirection() const {
	return _threshold_reset_dimension;
}

unsigned int Mesh::getGridThresholdResetJumpDirection() const {
	return _threshold_reset_jump_dimension;
}

unsigned int Mesh::getGridNumDimensions() const {
	return _grid_num_dimensions;
}

Mesh::GridCellTransition Mesh::calculateCellTransition(double efficacy, unsigned int dim) const{
	unsigned int offset = (unsigned int)abs(efficacy/this->getGridCellWidthByDimension(dim));
	double goes = (double)fabs(efficacy / this->getGridCellWidthByDimension(dim)) - offset;
	double stays = 1.0 - goes;

	int offset_1 = efficacy > 0 ? -offset : offset;
	int offset_2 = efficacy > 0 ? -(offset+1) : -(offset-1);

	return Mesh::GridCellTransition(stays, goes, offset_1, offset_2);
}

void Mesh::CreateCells(){
	// This function is called in the streamline (Python) version of a mesh file.
	// All Cells are in fact quadrilaterals here
	vector<double> v_points(Quadrilateral::_nr_points);
	vector<double> w_points(Quadrilateral::_nr_points);

	int i = 0;

	// Create a strip zero without Quadrilaterals to maintain correspondence to the Python numbering
	// scheme which starts with strip one.
	vector<Cell> vec_quad;
	_vec_vec_quad.push_back(vec_quad);

	for( auto it = _vec_block.begin(); it != _vec_block.end(); it++, i++){
		// loop over all elements except for the last
		const vector<vector<double> >& vec_v = it->_vec_v;
		const vector<vector<double> >& vec_w = it->_vec_w;


		// loop over all strips points, but not the last one
		int strip = 0;
		for(auto strit = vec_v.begin(); strit != --vec_v.end(); strit++, strip++){
			unsigned int m = std::min(strit->size(),(strit+1)->size());
			vector<Cell> vec_quad;
			for (unsigned cell = 0; cell < m-1; cell++){
				v_points[0] = vec_v[strip][cell];
				v_points[1] = vec_v[strip][cell+1];
				v_points[2] = vec_v[strip+1][cell+1];
				v_points[3] = vec_v[strip+1][cell];

				w_points[0] = vec_w[strip][cell];
				w_points[1] = vec_w[strip][cell+1];
				w_points[2] = vec_w[strip+1][cell+1];
				w_points[3] = vec_w[strip+1][cell];
				Cell quad(v_points, w_points);
				vec_quad.push_back(quad);
			}
			_vec_vec_quad.push_back(vec_quad);
		}
	}
}

void Mesh::CreateNeighbours(){
	unsigned int i = 0;

	for (auto it = _vec_vec_quad.begin(); it != _vec_vec_quad.end(); it++, i++){
		int j = 0;
		for (auto itcell = it->begin(); itcell != it->end(); itcell++,j++){
			for (auto it_point = itcell->Points().begin(); it_point != itcell->Points().end(); it_point++){
				const Point& p = *it_point;
				if (_map.count(p) == 0 ){
					// position index must point to the location that will just be added to the map
					_map[p] = _vec_vec_cell.size();
					vector<Coordinates> vec;
					vec.push_back(Coordinates(i,j));
					_vec_vec_cell.push_back(vec);
				} else {
					_vec_vec_cell[_map[p]].push_back(Coordinates(i,j));
				}
			}
		}
	}
}

void Mesh::InsertStationary(const Cell& quad){
	_vec_vec_quad[0].push_back(quad);
}

vector<Coordinates> Mesh::PointBelongsTo(const Point& p) const {
	if (_map.count(p) == 1){
		unsigned int ind  = _map.at(p);
		return _vec_vec_cell[ind];
	}
	else
		throw TwoDLibException("Position does not exist in Mesh");
}

vector<Coordinates> Mesh::findV(double V, Threshold th) const
{
	vector<Coordinates> vec_ret;
	int i = 0;
	for (auto it = _vec_vec_quad.begin(); it != _vec_vec_quad.end(); it++, i++){
		int j = 0;
		for (auto itcell = it->begin(); itcell != it->end(); itcell++,j++){

			double V_min =  std::numeric_limits<double>::max();
			double V_max = -std::numeric_limits<double>::max();
			for (auto itpoint = itcell->Points().begin(); itpoint != itcell->Points().end(); itpoint++){
				if ( (*itpoint)[0] > V_max )
					V_max = (*itpoint)[0];
				if ( (*itpoint)[0] < V_min )
					V_min = (*itpoint)[0];
			}

			if (th == EQUAL)
				if (V >= V_min && V <= V_max)
					vec_ret.push_back(Coordinates(i,j));
			if (th == ABOVE)
				if (V < V_min)
					vec_ret.push_back(Coordinates(i,j));
			if (th == BELOW)
				if (V > V_max)
					vec_ret.push_back(Coordinates(i,j));
		}
	}
	return vec_ret;
}

bool Mesh::stripsAreVOriented() const {
	return _strips_are_v_oriented;
}

vector<Coordinates> Mesh::allCoords() const{
	vector<Coordinates> vec_ret;
	for (unsigned int i = 0; i < _vec_vec_quad.size(); i++){
		for (unsigned int j = 0; j < _vec_vec_quad[i].size(); j++){
			vec_ret.push_back(Coordinates(i,j));
		}
	}

	return vec_ret;
}

vector<Coordinates> Mesh::findPointInMeshSlow(const Point& p, const double u) const{
	vector<Coordinates> vec_ret;
	for (unsigned int i = 0; i < _vec_vec_quad.size(); i++){
		for (unsigned int j = 0; j < _vec_vec_quad[i].size(); j++){
			if ( _vec_vec_quad[i][j].IsInside(p))
				vec_ret.push_back(Coordinates(i,j));
		}
	}

	if (vec_ret.size() == 0) { // uh oh, maybe this is a grid, not a mesh
		if (_grid_num_dimensions >= 3) {
			if (p[0] < getGridBaseByDimension(_grid_num_dimensions - 1) + getGridSizeByDimension(_grid_num_dimensions - 1) &&
				p[0] > getGridBaseByDimension(_grid_num_dimensions - 1) &&
				p[1] < getGridBaseByDimension(_grid_num_dimensions - 2) + getGridSizeByDimension(_grid_num_dimensions - 2) &&
				p[1] > getGridBaseByDimension(_grid_num_dimensions - 2) &&
				u < getGridBaseByDimension(_grid_num_dimensions - 3) + getGridSizeByDimension(_grid_num_dimensions - 3) &&
				u > getGridBaseByDimension(_grid_num_dimensions - 3)) {

				unsigned int i = int(((p[0] - getGridBaseByDimension(_grid_num_dimensions - 1)) / getGridSizeByDimension(_grid_num_dimensions - 1)) * getGridResolutionByDimension(_grid_num_dimensions - 1));
				unsigned int j = int(((p[1] - getGridBaseByDimension(_grid_num_dimensions - 2)) / getGridSizeByDimension(_grid_num_dimensions - 2)) * getGridResolutionByDimension(_grid_num_dimensions - 2));
				unsigned int k = int(((u - getGridBaseByDimension(_grid_num_dimensions - 3)) / getGridSizeByDimension(_grid_num_dimensions - 3)) * getGridResolutionByDimension(_grid_num_dimensions - 3));

				unsigned int strips = j + (k * getGridResolutionByDimension(_grid_num_dimensions - 3));

				vec_ret.push_back(Coordinates(strips, i));
			}
			else {
				throw TwoDLibException("Position does not exist in 3D Grid");
			}
		}
		else {
			if (p[0] < getGridBaseByDimension(_grid_num_dimensions - 1) + getGridSizeByDimension(_grid_num_dimensions - 1) &&
				p[0] > getGridBaseByDimension(_grid_num_dimensions - 1) &&
				p[1] < getGridBaseByDimension(_grid_num_dimensions - 2) + getGridSizeByDimension(_grid_num_dimensions - 2) &&
				p[1] > getGridBaseByDimension(_grid_num_dimensions - 2)) {

				unsigned int i = int(((p[0] - getGridBaseByDimension(_grid_num_dimensions - 1)) / getGridSizeByDimension(_grid_num_dimensions - 1)) * getGridResolutionByDimension(_grid_num_dimensions - 1));
				unsigned int j = int(((p[1] - getGridBaseByDimension(_grid_num_dimensions - 2)) / getGridSizeByDimension(_grid_num_dimensions - 2)) * getGridResolutionByDimension(_grid_num_dimensions - 2));

				vec_ret.push_back(Coordinates(j, i));
			}
			else {
				throw TwoDLibException("Position does not exist in Grid");
			}
		}
		
	}

	return vec_ret;
}

void Mesh::ProcessFileIntoBlocks(std::ifstream& ifst){

	string line;
	Mesh::Block block;

	//absorb the first two lines, which contain meta description, the first one
	// has already been eaten in the XML test
	getline(ifst,line);
	_t_step = boost::lexical_cast<double>(line);
	unsigned int count = 0;

	bool ending = false;

	// now start parsing
	while ( getline(ifst, line ) ) {
		std::istringstream is( line );
		if (is.str() == "end"){ending = true; break;}
		if (is.str() != string("closed") ){
			vector<double> vec=std::vector<double>( std::istream_iterator<double>(is),std::istream_iterator<double>() );
			if (count%2 == 0)
				block._vec_v.push_back(vec);
			else
				block._vec_w.push_back(vec);
			count++;
		}
		else {
			count = 0;
			_vec_block.push_back(block);
			block._vec_v.clear();
			block._vec_w.clear();
		}
	}
	if (!ending)
		throw TwoDLibException("Mesh file not closed properly");
	if (count != 0) _vec_block.push_back(block);

}

void Mesh::GeneratePoints(Coordinates c, vector<Point>* pvec)
{
	assert(c[0] >= 1);
	_vec_vec_gen[c[0]][c[1]].Generate(pvec);
}

std::vector<Coordinates> Mesh::CellsBelongTo(const Cell& quad) const
{
	std::vector<Coordinates> vec_ret;
	for (unsigned int i = 0; i < _vec_vec_quad.size(); i++)
		for (unsigned int j = 0; j < _vec_vec_quad[i].size(); j++){
			for (auto it = _vec_vec_quad[i][j].Points().begin(); it != _vec_vec_quad[i][j].Points().end(); it++)
				if (quad.IsInside(*it))
					vec_ret.push_back(TwoDLib::Coordinates(i,j));
		}
	return vec_ret;
}

void Mesh::FillTimeFactor()
{
	this->_vec_timefactor = std::vector<unsigned int>(_vec_vec_quad.size(),1);
}

bool Mesh::ProcessNonXML(std::ifstream& ifst)
{
	// not XML
	this->ProcessFileIntoBlocks(ifst);
	this->CreateCells();
	this->CreateNeighbours();
	this->FillTimeFactor();

	return true;
}

Mesh::Mesh
(
	const string&  file_name
):
_vec_block(0),
_vec_vec_quad(0),
_vec_vec_gen(0)
{
	std::ifstream ifst(file_name);
	if (!ifst){
		std::cerr << "Can't open mesh file." << std::endl;
		throw TwoDLibException("Can't open mesh file.");
	}
	else {
		string line;
		getline(ifst,line);
		// in some hand edits the Mesh tag is not place at the beginning; this is valid XML and the parser is not bothered by it,
		// but it must be passed correctly to the XML:
		line.erase (std::remove (line.begin(), line.end(), ' '), line.end());

		// instead of string comparison, merely find if <Mesh> is substring of the first line
		// MdK: 23/10/2017
		// accept Model as tag;
		// MdK: 04/12/2017
		if (line.find("<Mesh>") != std::string::npos || line.find("<Model>") != std::string::npos){
			ifst.close();
			std::ifstream newifst(file_name);
			this->FromXML(newifst);
		}
		else {
			this->ProcessNonXML(ifst);
		}
	}
}

bool Mesh::CheckAreas() const {

	for (unsigned int i = 0; i < _vec_vec_quad.size(); i++)
		for (unsigned int j = 0; j < _vec_vec_quad[i].size(); j++)
			if (_vec_vec_quad[i][j].SignedArea() == 0){
				return false;
			}
	return true;
}

std::vector<Cell> Mesh::FromVals(const std::vector<double>& vals) const
{
	// this function should only be called on python meshes as it makes
	// assumptions that cells are Quadrilaterals
	assert(vals.size()%8 == 0);
	vector<Cell> vec_ret;

	unsigned int  nr_chuncks = vals.size()/8;

	vector<double> vs(Quadrilateral::_nr_points);
	vector<double> ws(Quadrilateral::_nr_points);
	for (unsigned int i = 0; i < nr_chuncks; i++){
		vs[0] = vals[i*8];
		vs[1] = vals[i*8 + 2];
		vs[2] = vals[i*8 + 4];
		vs[3] = vals[i*8 + 6];
		ws[0] = vals[i*8 + 1];
		ws[1] = vals[i*8 + 3];
		ws[2] = vals[i*8 + 5];
		ws[3] = vals[i*8 + 7];

		Cell quad(vs, ws);
		vec_ret.push_back(quad);
	}

	return vec_ret;
}

unsigned int Mesh::TimeFactorFromStrip(const pugi::xml_node& node) const
{
	// find whether the timefactor attribute is there. if not time factor is 1.
	// if it is, it is whatever the attribute values says it is

	int tf = node.attribute("timefactor").as_int();
	if (tf == 0) tf = 1;

	return tf;
}

std::vector<double> Mesh::StripValuesFromStream(std::istream& s) const
{
	vector<double> cs;

	std::string token;
	while (getline(s, token,' ')){
		if (!token.empty())
			cs.push_back(std::stod(token));
    }
	return cs;
}

std::vector<TwoDLib::Cell> Mesh::CellsFromXMLStrip(const pugi::xml_node& strip, unsigned int time_factor) const
{

	std::istringstream ivals(strip.first_child().value());
    vector<double> cs = this->StripValuesFromStream(ivals);
    // Empty strips are legal
    if (cs.size() == 0)
    	return std::vector<TwoDLib::Cell>(0);

    int n_coords = 4*(time_factor + 1);
	if (cs.size()%n_coords != 0 )
		throw TwoDLibException("Unexpected number of points in strip during XML read");

	std::vector<TwoDLib::Cell> vec_ret = this->CellsFromValues(cs,n_coords);

	return vec_ret;

}

std::vector<TwoDLib::Cell> Mesh::CellsFromValues(const std::vector<double>& vec_vals, unsigned int n_coords) const
{
	// n_coords is the number of coordinate values, so double the number of points.
	// the number of cells is then vec_vals.size()/n_coords

	std::vector<TwoDLib::Cell> vec_cells;
	unsigned int n_cells = vec_vals.size()/n_coords;

	for (unsigned int icell = 0; icell < n_cells; icell++)
	{
		std::vector<double> vs;
		std::vector<double> ws;
		unsigned int n_points = n_coords/2;
		for(int ipoint = 0; ipoint < n_points; ipoint++)
		{
			vs.push_back(vec_vals[icell*n_coords + 2*ipoint]);
			ws.push_back(vec_vals[icell*n_coords + 2*ipoint + 1]);
		}
		Cell cell(vs,ws);
		vec_cells.push_back(cell);
	}
	return vec_cells;
}


void Mesh::FromXML(istream& s)
{
    pugi::xml_document doc;
    pugi::xml_parse_result result;

    result = doc.load(s);
    if (!result)
    	throw TwoDLib::TwoDLibException("Couldn't parse Mesh from stream");

    pugi::xml_node node_mesh = doc.first_child();
    if  (node_mesh.name() == std::string("Model"))
    	node_mesh = doc.first_child().child("Mesh");

    // Extract time step
    pugi::xml_node node_ts = node_mesh.child("TimeStep");
    if (! node_ts)
    	throw TwoDLib::TwoDLibException("Couldn't identify time step in mesh");
    std::istringstream ival(node_ts.first_child().value());
	ival >> _t_step;

	_grid_num_dimensions = 2;
	//Extract number of dimensions if it's there.
	node_ts = node_mesh.child("GridNumDimensions");
	if (node_ts) {
		std::istringstream ival(node_ts.first_child().value());
		ival >> _grid_num_dimensions;
	}

	_resolution = vector<unsigned int>(_grid_num_dimensions);
	_dimensions = vector<double>(_grid_num_dimensions);
	_base = vector<double>(_grid_num_dimensions);

	node_ts = node_mesh.child("GridDimensions");
	if (node_ts) {
		std::istringstream ival(node_ts.first_child().value());
		for (unsigned int i = 0; i < _grid_num_dimensions; i++) {
			ival >> _dimensions[i];
		}
	}

	node_ts = node_mesh.child("GridResolution");
	if (node_ts) {
		std::istringstream ival(node_ts.first_child().value());
		for (unsigned int i = 0; i < _grid_num_dimensions; i++) {
			ival >> _resolution[i];
		}
	}
	
	node_ts = node_mesh.child("GridBase");
	if (node_ts) {
		std::istringstream ival(node_ts.first_child().value());
		for (unsigned int i = 0; i < _grid_num_dimensions; i++) {
			ival >> _base[i];
		}
	}

	// For now, just say that the direction of the strip is the threshold reset direction
	// TODO: Allow this to be overridden in the model XML
	_threshold_reset_dimension = _grid_num_dimensions - 1;
	_threshold_reset_jump_dimension = _grid_num_dimensions - 2;
	_strips_are_v_oriented = true;
	_has_defined_strips = false;

	if (_resolution[0] == 0) { // this is a mesh or an old style grid where the resolution isn't in the XML
		_num_strips = 0;
		for (pugi::xml_node strip = node_mesh.child("Strip"); strip; strip = strip.next_sibling("Strip")) {

			unsigned int time_factor = this->TimeFactorFromStrip(strip);
			_vec_timefactor.push_back(time_factor);
			vector<Cell> vec_quad = this->CellsFromXMLStrip(strip, time_factor);
			_vec_vec_quad.push_back(vec_quad);
			_num_strips++;
		}

		_has_defined_strips = true;

		// We're unfortunately still potentially using the old way of doing 2D
		// i.e strips in a different direction instead of always horizontal.
		// We need to do work to identify which direction is which.

		// Start by setting values assuming v is horizontal
		double min_v = _vec_vec_quad[0][0].getVecV()[0];
		double max_v = _vec_vec_quad[0][_vec_vec_quad[0].size() - 1].getVecV()[2];

		double min_w = _vec_vec_quad[0][0].getVecW()[0];
		double max_w = _vec_vec_quad[_vec_vec_quad.size() - 1][0].getVecW()[3];


		Quadrilateral q1 = Quad(1, 0);
		Quadrilateral q2 = Quad(1, 1);

		double horiz_dist = std::fabs(q2.Centroid()[0] - q1.Centroid()[0]);
		double vert_dist = std::fabs(q2.Centroid()[1] - q1.Centroid()[1]);

		// one of these distances should be close to zero, so pick the other one
		// we do this because we don't know if this is a v- or h- efficacy

		if (vert_dist > horiz_dist) { // This is the v distance
			min_v = _vec_vec_quad[0][0].getVecW()[0];
			max_v = _vec_vec_quad[0][_vec_vec_quad[0].size() - 1].getVecW()[2];

			min_w = _vec_vec_quad[0][0].getVecV()[0];
			max_w = _vec_vec_quad[_vec_vec_quad.size() - 1][0].getVecV()[3];

			_threshold_reset_dimension = _grid_num_dimensions - 2;
			_threshold_reset_jump_dimension = _grid_num_dimensions - 1;

			_strips_are_v_oriented = false;
		}

		_resolution[1] = _vec_vec_quad[0].size();
		_resolution[0] = _num_strips;

		_base[1] = min_v;
		_base[0] = min_w;

		_dimensions[1] = max_v - _base[1];
		_dimensions[0] = max_w - _base[0];

		this->CreateNeighbours();
	}
	else {
		_strip_length = _resolution[_grid_num_dimensions - 1];
		_num_strips = 1;
		for (unsigned int d = 0; d < _resolution.size() - 1; d++) { _num_strips *= _resolution[d]; }
	}
}

void Mesh::ToXML(ostream& s) const{
	s << std::setprecision(14);
	s << "<Mesh>\n";
	s << "<TimeStep>" << this->TimeStep() << "</TimeStep>\n";
	for(unsigned int i = 0; i < _vec_vec_quad.size(); i++){
		s << "<Strip>";
		for(unsigned int j = 0; j < _vec_vec_quad[i].size(); j++){
			for (const Point& p: _vec_vec_quad[i][j].Points()){
				s << p[0] << " " << p[1] << " ";
			}
		}
		s << "</Strip>\n";
	}
	s << "</Mesh>\n";
}
