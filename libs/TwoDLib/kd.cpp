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
#include <cstring>
#include <vector>
#include "kd.h"


/* see quickselect method */
    TwoDLib::kd_node_t* find_median(TwoDLib::kd_node_t *start, TwoDLib::kd_node_t *end, int idx)
{
    if (end <= start) return NULL;
    if (end == start + 1)
        return start;

    TwoDLib::kd_node_t *p, *store, *md = start + (end - start) / 2;
    double pivot;
    while (1) {
        pivot = md->x[idx];

        swap(md, end - 1);
        for (store = p = start; p < end; p++) {
            if (p->x[idx] < pivot) {
                if (p != store)
                    swap(p, store);
                store++;
            }
        }
        swap(store, end - 1);

        /* median has duplicate values */
        if (store->x[idx] == md->x[idx])
            return md;

        if (store > md) end = store;
        else        start = store;
    }
}

    TwoDLib::kd_node_t* TwoDLib::make_tree(void* buffer, TwoDLib::kd_node_t *t, int len, int i, int dim)
    {
//    	TwoDLib::kd_node_t *n;
    	kd_node_t* n = new(buffer) kd_node_t;
    	if (!len) return 0;

    	if ((n = find_median(t, t + len, i))) {
    		i = (i + 1) % dim;
    		n->left  = make_tree(buffer, t, n - t, i, dim);
    		n->right = make_tree(buffer, n + 1, t + len - (n + 1), i, dim);
    	}
    	return n;
    }


void  TwoDLib::nearest(TwoDLib::kd_node_t *root, TwoDLib::kd_node_t *nd, int i, int dim,
        TwoDLib::kd_node_t **best, double *best_dist, int* visited, double d_rad, std::vector<Point>* p_near)
{
    double d, dx, dx2;

    if (!root) return;
    d = dist(root, nd, dim);
    dx = root->x[i] - nd->x[i];
    dx2 = dx * dx;


    if (d_rad > 0 && dim == 2 && dx2 < d_rad)
    	p_near->push_back(Point(root->x[0],root->x[1]));

    (*visited)++;

    if (!*best || d < *best_dist) {
        *best_dist = d;
        *best = root;
    }

    /* if chance of exact match is high */
    if (!*best_dist) return;

    if (++i >= dim) i = 0;

    nearest(dx > 0 ? root->left : root->right, nd, i, dim, best, best_dist, visited, d_rad, p_near);
    if (dx2 >= *best_dist) return;
    nearest(dx > 0 ? root->right : root->left, nd, i, dim, best, best_dist, visited, d_rad, p_near);
}
