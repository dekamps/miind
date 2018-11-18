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


#ifndef _CODE_LIBS_TWODLIB_TRANSITIONMATRIXGENERATOR_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_TRANSITIONMATRIXGENERATOR_INCLUDE_GUARD

#include <vector>
#include "FiducialElement.hpp"
#include "MeshTree.hpp"
#include "Uniform.hpp"


using std::vector;

namespace TwoDLib {

	//! Default argument for TransitionMatrixGenerator
    const FidElementList list;


  /**
   * \brief Generates a transition matrix for a given mesh, and a given translation
   *
   *
   */

  class TransitionMatrixGenerator {
  public:
    	//! A translated point may end up outside of the mesh. If it is Lost, it is unaccounted for. If
    	//! it is Accounted, it ended up in fidicual volume, but not in any of the Mesh cells. If it is found, it ended up in one of the Mesh cells.
    	enum SearchResult {Lost = -2, Accounted = -1, Found = 0 };
		struct Hit {
			  Coordinates	_cell;
			  int 			_count;
        double    _prop;

			  Hit():_count(Lost),_prop(0){};
		  };

	  //! A TransitionMatrixGenerator. It accepts a MeshTree (effectively a Mesh proxy, a reference to a
	  //! random generator, and a number of events used in the Monte Carlo calculation of transition matrix
	  //! elements. It is possible to add fidicual Quadrilaterals. If translated Monte Carlo fall outside the
	  //! grid without fiducial elements, they are lost. This may be acceptable. If not construct fiducial
	  //! Quadrilateral that partly overlap with Mesh, but cover an area outside that will cover all generated
	  //! stray events. This will also reduce computation time a bit, because the fidcucial areas will be checked
	  //! first.
	  TransitionMatrixGenerator
	  (
		const MeshTree&, 	 			 // kdtree constructed from a Mesh
		Uniform&, 			 			 // reference to the random generator that is in common use
		unsigned int N = 10, 			 // number of points used in the MC generation for the calculation of elements
		const FidElementList& l = list
		);

	  //! Specify which transitions must be generated
	  void GenerateTransition
	  	  (
			  unsigned int,	//!< strip no
			  unsigned int,	//!< cell no
			  double,		//!< translation in the v direction
			  double    	//!< translation in the w direction
	  	  );

    //! Specify which transitions must be generated
	  void GenerateTransitionUsingQuadTranslation
	  	  (
			  unsigned int,	//!< strip no
			  unsigned int,	//!< cell no
			  double,		//!< translation in the v direction
			  double,    	//!< translation in the w direction
        std::vector<Coordinates>  //!< above threshold cells
	  	  );

    void GenerateTransformUsingQuadTranslation
    (
      unsigned int,
      unsigned int,
      const TwoDLib::MeshTree&,
      std::vector<Coordinates>);

	  //! After a simulation, the generator must be reset
	  void Reset(unsigned int N = 10);

	  //! List of points not attributed to any cell
	  vector<Point> LostPoints() const { return _lost; }

	  //! List of points attributed to a fiducial volume, but not a cell
	  vector<Point> AccountedPoints() const { return _accounted; }

	  //! List of cells that are hit by a Monte Carlo point, together with hit count
	  vector<Hit>   HitList() const { return _hit_list; }

	  //! Number of points used in MC generation
	  unsigned int N() const {return _N;}

  private:

	  bool CheckHitList				(const Coordinates&);
	  void ApplyTranslation			(vector<Point>*, const Point&);
	  double DetermineDistance		(const Quadrilateral&);
	  void ProcessTranslatedPoints	(const vector<Point>& vec);
	  bool IsInAssociated			(const FiducialElement&, const Point&, Coordinates*);
	  SearchResult LocatePoint		(const Point&, Coordinates*);
	  Hit CheckTree					(const Point&, const vector<Point>&);
	  SearchResult CheckFiducial	(const Point&, Coordinates*);

	  vector<FiducialElement> InitializeFiducialVector(const Mesh&, const FidElementList&) const;

	  static double scale_distance;

	  const MeshTree&  		_tree;
	  Uniform&				_uni;
	  unsigned int 			_N;

    Point _grid_bottom_left;
    Point _grid_extent;
    bool _grid_normal_orientation;

	  std::vector<Hit> 		_hit_list;
	  std::vector<Point>	_lost;
	  std::vector<Point>	_accounted;

	  std::vector<FiducialElement>	_vec_fiducial;
  };
}

#endif
