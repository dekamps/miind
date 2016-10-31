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

Mesh::Mesh(istream& s):
_vec_block(0),
_vec_vec_quad(0),
_vec_vec_gen(0){
	this->FromXML(s);
}

Mesh::Mesh(const Mesh& m):
_vec_block(m._vec_block),
_vec_vec_quad(m._vec_vec_quad),
_vec_vec_gen(m._vec_vec_gen),
_t_step(m._t_step),
_map(m._map),
_vec_vec_cell(m._vec_vec_cell)
{
}


void Mesh::CreateCells(){
	vector<double> v_points(Quadrilateral::_nr_points);
	vector<double> w_points(Quadrilateral::_nr_points);

	int i = 0;

	// Create a strip zero without Quadrilaterals to maintain correspondence to the Python numbering
	// scheme which starts with strip one.
	vector<Quadrilateral> vec_quad;
	_vec_vec_quad.push_back(vec_quad);

	for( auto it = _vec_block.begin(); it != _vec_block.end(); it++, i++){
		// loop over all elements except for the last
		const vector<vector<double> >& vec_v = it->_vec_v;
		const vector<vector<double> >& vec_w = it->_vec_w;


		// loop over all strips points, but not the last one
		int strip = 0;
		for(auto strit = vec_v.begin(); strit != --vec_v.end(); strit++, strip++){
			unsigned int m = std::min(strit->size(),(strit+1)->size());
			vector<Quadrilateral> vec_quad;
			for (unsigned cell = 0; cell < m-1; cell++){
				v_points[0] = vec_v[strip][cell];
				v_points[1] = vec_v[strip][cell+1];
				v_points[2] = vec_v[strip+1][cell+1];
				v_points[3] = vec_v[strip+1][cell];

				w_points[0] = vec_w[strip][cell];
				w_points[1] = vec_w[strip][cell+1];
				w_points[2] = vec_w[strip+1][cell+1];
				w_points[3] = vec_w[strip+1][cell];
				Quadrilateral quad(v_points, w_points);
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

void Mesh::InsertStationary(const Quadrilateral& quad){
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

std::vector<Coordinates> Mesh::CellsBelongTo(const Quadrilateral& quad) const
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
		if (line == string("<Mesh>")){
			ifst.close();
			std::ifstream newifst(file_name);
			this->FromXML(newifst);
		}
		else {
			// not XML
			this->ProcessFileIntoBlocks(ifst);
			this->CreateCells();
			this->CreateNeighbours();
		}
	}
	if (! this->CheckAreas() )
		throw TwoDLib::TwoDLibException("Zero area in mesh.");
}

bool Mesh::CheckAreas() const {

	for (unsigned int i = 0; i < _vec_vec_quad.size(); i++)
		for (unsigned int j = 0; j < _vec_vec_quad[i].size(); j++)
			if (_vec_vec_quad[i][j].SignedArea() == 0)
				return false;
	return true;
}

std::vector<Quadrilateral> Mesh::FromVals(const std::vector<double>& vals) const
{
	assert(vals.size()%8 == 0);
	vector<Quadrilateral> vec_ret;

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

		Quadrilateral quad(vs, ws);
		vec_ret.push_back(quad);
	}

	return vec_ret;
}

void Mesh::FromXML(istream& s)
{
    pugi::xml_document doc;
    pugi::xml_parse_result result;

    result = doc.load(s);
    if (!result)
    	throw TwoDLib::TwoDLibException("Couldn't parse Mesh from stream");

    pugi::xml_node node_mesh = doc.first_child();

    // Extract time step
    pugi::xml_node node_ts = node_mesh.child("TimeStep");
    if (! node_ts)
    	throw TwoDLib::TwoDLibException("Couldn't identify time step in mesh");
    std::istringstream ival(node_ts.first_child().value());
	ival >> _t_step;


    for (pugi::xml_node strip = node_mesh.child("Strip"); strip; strip = strip.next_sibling("Strip")){

		vector<double> cs;
		std::istringstream ivals(strip.first_child().value());
		while (ivals){
			double d;
			ivals >> d;
			if (ivals.good())
				cs.push_back(d);
		}
		if (cs.size()%8 != 0 )
			throw TwoDLibException("Unexpected number of points in strip during XML read");

		vector<Quadrilateral> vec_quad = FromVals(cs);
		_vec_vec_quad.push_back(vec_quad);
    }

	// build the mapping and the list of lists that allows to tell which cells a mesh point belongs to
	this->CreateNeighbours();
}

void Mesh::ToXML(ostream& s) const{
	s << std::setprecision(10);
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
