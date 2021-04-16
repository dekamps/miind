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

#ifndef _CODE_LIBS_GEOMLIB_MESH_INCLUDE_GUARD
#define _CODE_LIBS_GEOMLIB_MESH_INCLUDE_GUARD

#include <string>
#include <unordered_map>
#include <functional>
#include "Coordinates.hpp"
#include "Uniform.hpp"
#include "PolyGenerator.hpp"
#include "pugixml.hpp"

using std::string;

class ifstream;

namespace TwoDLib {

	//* Representation of a two dimensional grid for storage of probability mass.
	class Mesh {
	public:


		typedef pair<Coordinates,unsigned int> CellCount;

		//!< Mesh can read two formats, the Python format, which is very specific, see documentation,
		//!< or the XML format. The XML format for a Mesh is not truly XML: each line starting with <Strip>
		//!< and ending with </Strip>object requires that a single strip is represented on a single line
		//!< and that no two strips are on the same line. In other words, no newline character may
		//!< occur between <Strip> and </Strip> and </Strip> must be followed by a new line character.
		//!< Files produced by mesh.py
		//!< or MatrixGenerator conform to this requirement.

		Mesh(double width) : _grid_cell_width(width) {}

		//!< construction from a disk representation via stream. Beware: The disk representation must be in XML format
		Mesh
		(
			std::istream& //!< input stream
		);

		//!< path name to file containing  disk representation of the Mesh. The file may be in XML or in Python format.
		Mesh
		(
			const string& //!< filename of disk representation mesh
		);

		//! copy constructor
		Mesh(const Mesh&);


		//!< destructor
		~Mesh(){}

		//!< number of strips in the grid
		virtual unsigned int NrStrips() const { return _vec_vec_quad.size(); }

		//!< number of cells in strip i
		virtual unsigned int NrCellsInStrip(unsigned int i) const { assert( i < _vec_vec_quad.size()); return _vec_vec_quad[i].size();}

		virtual const Cell& Quad(unsigned int i, unsigned int j) const { return _vec_vec_quad[i][j]; }

		//!< Provide a mesh point, the function returns a list of Coordinates that this point belongs to
		//!< Caution! Stationary points will not show up in the returned list and must be tested separately
		vector<Coordinates> PointBelongsTo(const Point&) const;

		//!<  Provide a list of cells that fall partly within this Quadrilateral
		vector <Coordinates> CellsBelongTo(const Cell&) const;

		//! Coordinates are in the 'Python' numbering scheme: strips are numbered from 1 upwards
		void GeneratePoints(Coordinates,vector<Point>*);

		friend class MeshTree;

		enum Threshold {ABOVE, EQUAL, BELOW };

		vector<Coordinates> allCoords() const;

		vector<Coordinates> findV(double V, Threshold) const;

		vector<Coordinates> findPointInMeshSlow(const Point&) const;

		//! These cells are labeled with Coordinates(0,j). They have no neighbours, and tests as to whether
		//! points fall inside them should be made directly; they can not be expected to show up in
		//! the results of PointBelongsTo. TtranslationMatrixGenerators will handle them correctly,
		//! as long as this method is called before the TranslationMatrixGenerator is associated with the Mesh.
		void  InsertStationary(const Cell&);

		//! Time step used in Mesh generation.
		double TimeStep() const {return _t_step;}

		//! Write to an XM format. It is guaranteed that a single strip is written on a single line,
		//! and therefore can be processed by getline
		void ToXML(std::ostream&) const;

		// Get the grid cell width (only useful if the mesh is a grid)
		double getCellWidth() const;
		double getCellHeight() const;

		double getVWidth() const;
		double getHHeight() const;

		double getGridMinV() const;
		double getGridMinH() const;

		unsigned int getGridResV() const;
		unsigned int getGridResH() const;

		class GridCellTransition{
		public:
			double _stays;
			double _goes;
			int _offset_1;
			int _offset_2;

			GridCellTransition(double s, double g, int o1, int o2):
			_stays(s), _goes(g), _offset_1(o1), _offset_2(o2) {}
		};

		GridCellTransition calculateCellTransition(double efficacy) const;

	protected:

		void FromXML(std::istream&);
		std::vector<Cell> FromVals(const std::vector<double>&) const;

		bool CheckAreas() const;

		void ProcessFileIntoBlocks(std::ifstream&);
		void CreateCells();
		void CreateNeighbours();
		void FillTimeFactor();

		std::vector<TwoDLib::Cell> CellsFromXMLStrip(const pugi::xml_node&, unsigned int) const;
		std::vector<double> StripValuesFromStream(std::istream&) const;
		std::vector<TwoDLib::Cell> CellsFromValues(const std::vector<double>&, unsigned int) const;

		unsigned int TimeFactorFromStrip(const pugi::xml_node&) const;

		bool ProcessNonXML(std::ifstream&); // in non xml, all cells are guaranteed to be quadrilateral

		struct Block {
			//!< Each block has a list of v_lists and of w_list, both of equal length
			vector< vector<double> > _vec_v;
			vector< vector<double> > _vec_w;

		};

		struct hash_position {
			size_t operator()(const Point& pos) const {
				return std::hash<double>()(pos[0]) ^ std::hash<double>()(pos[1]);
			}
		};

		static const int					_dimension; // we work with two dimensional points

		vector<Block>                    	_vec_block;
		vector<vector<Cell> >   			_vec_vec_quad;
		vector<vector<PolyGenerator> >	    _vec_vec_gen;
		vector<unsigned int>              	_vec_timefactor;
		double								_t_step;
		double										_grid_cell_width;
		double										_grid_cell_height;

		double										_grid_v_width; // this is different to _grid_cell_width because it is independent of the direction of the efficacy
		double										_grid_h_height;
		double										_grid_min_v;
		double										_grid_min_h;
		unsigned int								_grid_res_v;
		unsigned int								_grid_res_h;

		// It is sometimes necessary to find out to which cells a given mesh point belongs.
		// A mesh point will be mapped to an index position in a list of a list of coordinates.
		// The list of coordinates at that position is the list of cells that this point belongs to.
		std::unordered_map<Point, unsigned int, hash_position>
													_map;   // hash map from position into vector of CellIds
		vector<vector<Coordinates> >				_vec_vec_cell;
	};

}

#endif // include guard
