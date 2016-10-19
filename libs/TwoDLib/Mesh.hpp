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
#include "QuadGenerator.hpp"

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
		unsigned int NrQuadrilateralStrips() const { return _vec_vec_quad.size(); }

		//!< number of cells in strip i
		unsigned int NrCellsInStrip(unsigned int i) const { assert( i < _vec_vec_quad.size()); return _vec_vec_quad[i].size();}

		const Quadrilateral& Quad(unsigned int i, unsigned int j) const { return _vec_vec_quad[i][j]; }

		//!< Provide a mesh point, the function returns a list of Coordinates that this point belongs to
		//!< Caution! Stationary points will not show up in the returned list and must be tested separately
		vector<Coordinates> PointBelongsTo(const Point&) const;

		//!<  Provide a list of cells that fall partly within this Quadrilateral
		vector <Coordinates> CellsBelongTo(const Quadrilateral&) const;

		//! Coordinates are in the 'Python' numbering scheme: strips are numbered from 1 upwards
		void GeneratePoints(Coordinates,vector<Point>*);

		friend class MeshTree;

		enum Threshold {ABOVE, EQUAL, BELOW };

		vector<Coordinates> findV(double V, Threshold) const;

		//! These cells are labeled with Coordinates(0,j). They have no neighbours, and tests as to whether
		//! points fall inside them should be made directly; they can not be expected to show up in
		//! the results of PointBelongsTo. TtranslationMatrixGenerators will handle them correctly,
		//! as long as this method is called before the TranslationMatrixGenerator is associated with the Mesh.
		void  InsertStationary(const Quadrilateral&);

		//! Time step used in Mesh generation.
		double TimeStep() const {return _t_step;}

		//! Write to an XM format. It is guaranteed that a single strip is written on a single line,
		//! and therefore can be processed by getline
		void ToXML(std::ostream&) const;

	private:

		void FromXML(std::istream&);
		std::vector<Quadrilateral> FromVals(const std::vector<double>&) const;

		bool CheckAreas() const;

		void ProcessFileIntoBlocks(std::ifstream&);
		void CreateCells();
		void CreateNeighbours();

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

		static const int							_dimension; // we work with two dimensional points
		vector<Block>                    			_vec_block;
		vector<vector<Quadrilateral> >   			_vec_vec_quad;
		vector<vector<QuadGenerator> >	            _vec_vec_gen;
		double										_t_step;

		// It is sometimes necessary to find out to which cells a given mesh point belongs.
		// A mesh point will be mapped to an index position in a list of a list of coordinates.
		// The list of coordinates at that position is the list of cells that this point belongs to.
		std::unordered_map<Point, unsigned int, hash_position>
													_map;   // hash map from position into vector of CellIds
		vector<vector<Coordinates> >				_vec_vec_cell;
	};

}

#endif // include guard
