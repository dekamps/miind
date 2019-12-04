#ifndef _CODE_LIBS_TWODLIB_MESH_ND_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_MESH_ND_INCLUDE_GUARD

#include <string>
#include <unordered_map>
#include <functional>
#include "Coordinates.hpp"
#include "Uniform.hpp"
#include "PolyGenerator.hpp"
#include "pugixml.hpp"
#include "Mesh.hpp"

using std::string;

class ifstream;

namespace TwoDLib {

    class MeshNd : public Mesh {
        public:
            MeshNd(double t_step, unsigned int num_dimensions, std::vector<unsigned int> resolution, std::vector<double> dimensions, std::vector<double> base)
            :Mesh(0.0),
            _num_dimensions(num_dimensions),
            _resolution(resolution),
            _dimensions(dimensions),
            _base(base)
            {
                _strip_length = _resolution[_num_dimensions-1];
                _num_strips = 1;
                for (unsigned int d = 0; d< _resolution.size()-1; d++) { _num_strips *= _resolution[d]; }
                _grid_cell_width =  dimensions[_num_dimensions-1] / _strip_length;
                _t_step = t_step;

                for (unsigned int i =0; i<_num_strips; i++){
                    std::vector<Cell> strip;
                    for (unsigned int j=0; j< _strip_length; j++) {
                        // For now, just generate four 2D points for the last two
                        // demensions
                        double v_width = _dimensions[_num_dimensions-1] / _resolution[_num_dimensions-1];
                        double w_width = _dimensions[0] / (_resolution[0]);
                        double pv = v_width * j;
                        double pw = w_width * i;
                        double bv = _base[_num_dimensions-1];
                        double bw = _base[0];

                        std::vector<Point> ps;
                        ps.push_back(Point(bv+pv, bw+pw));
                        ps.push_back(Point(bv+pv+v_width, bw+pw));
                        ps.push_back(Point(bv+pv+v_width, bw+pw+w_width));
                        ps.push_back(Point(bv+pv, bw+pw+w_width));
                        
                        strip.push_back(Cell(ps));
                    }
                    _vec_vec_quad.push_back(strip);
                }
            }

            //!< number of strips in the grid
            unsigned int NrStrips() const override { return _num_strips; }

            //!< number of cells in strip i => In the ND Grid, all stips are the same length
            unsigned int NrCellsInStrip(unsigned int i) const override { return _strip_length; }

            const Cell& Quad(unsigned int i, unsigned int j) const override {
                return _vec_vec_quad[i][j];
            }

        private:
            unsigned int _num_dimensions;
            std::vector<unsigned int> _resolution;
            std::vector<double> _dimensions;
            std::vector<double> _base;

            unsigned int _num_strips;
            unsigned int _strip_length;
    };

}

#endif //_CODE_LIBS_TWODLIB_MESH_ND_INCLUDE_GUARD