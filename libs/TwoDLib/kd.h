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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

// This code was adapted from http://rosettacode.org/wiki/K-d_tree, and made available there under:
// http://www.gnu.org/licenses/fdl-1.2.html. Authors can be found in the history tab of that page.

#ifndef _CODE_LIBS_TWODLIB_KD_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_KD_INCLUDE_GUARD

#include <cstring>
#include <vector>
#include "Point.hpp"

namespace TwoDLib {

const unsigned int MAX_DIM = 3;

	struct kd_node_t{
    	double x[MAX_DIM];
    	struct kd_node_t *left, *right;
	};

	inline void swap(kd_node_t *x, kd_node_t *y) {
	    double tmp[MAX_DIM];
	    memcpy(tmp,  x->x, sizeof(tmp));
	    memcpy(x->x, y->x, sizeof(tmp));
	    memcpy(y->x, tmp,  sizeof(tmp));
	}

    inline double dist(kd_node_t *a, kd_node_t *b, int dim)
    {
    	double t, d = 0;
    	while (dim--) {
    		t = a->x[dim] - b->x[dim];
    		d += t * t;
    	}
    	return d;
    }

    kd_node_t* make_tree(void*, kd_node_t *t, int len, int i, int dim);

    void nearest(kd_node_t *root, kd_node_t *nd, int i, int dim,
           kd_node_t **best, double *best_dist, int *i_visted, double d_nearest = 0, std::vector<Point>* p = 0);

}

#endif // include guard
