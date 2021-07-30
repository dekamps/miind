/*
 * main.cpp
 *
 *  Created on: Jan 21, 2016
 *      Author: scsmdk
 */
#include <iostream>
#include <iterator>
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>
#include <iomanip>
#include "TwoDLib.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif

const int N_POINTS = 10;

namespace TwoDLib {

  inline void split(const string &s, char delim, vector<string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (getline(ss, item, delim)) {
      elems.push_back(item);
    }
  }


  inline std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<string> elems;
    split(s, delim, elems);
    return elems;
  }
}

std::pair< TwoDLib::Point, TwoDLib::Point> Analyse(const TwoDLib::Mesh& mesh){
  TwoDLib::Point ll( std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
  TwoDLib::Point ur(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
 //establish bounding box

  for( unsigned int i = 0; i < mesh.NrStrips(); i++ )
    for (unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++ ){
      const TwoDLib::Cell& quad = mesh.Quad(i,j);
      for (const TwoDLib::Point& p: quad.Points()){
	if (p[0] < ll[0] )
	  ll[0] = p[0];
        if (p[1] < ll[1] )
	  ll[1] = p[1];
        if (p[0] > ur[0] )
	  ur[0] = p[0];
	if (p[1] > ur[1])
	  ur[1] = p[1];
      }
    }

  return std::pair<TwoDLib::Point, TwoDLib::Point>(ll,ur);
}


unsigned int  Bin(double x,  double x_min, double x_max, unsigned int n_x){

  double binsize = (x_max  - x_min)/static_cast<float>(n_x);
  unsigned int bin = (int)floor((x - x_min)/binsize);
  if (bin == n_x)
    --bin;

  return bin;
}

void CalculateProjectionsGeometric(std::ofstream& ofst,
 const TwoDLib::Mesh& mesh,
 double v_min,
 double v_max,
 unsigned int nv,
 double w_min,
 double w_max,
 unsigned int nw
 ) {
  ofst << "<transitions>\n";
  
  std::vector<double> dimensions = {(w_max-w_min),(v_max-v_min)};
  std::vector<double> base = {w_min, v_min};
  std::vector<unsigned int> resolution = {nw, nv};
  
  TwoDLib::MeshNd bins(0.1, 2, resolution, dimensions, base);
  TwoDLib::MeshTree tree(mesh);
  TwoDLib::MeshTree tree_bins(bins);
  TwoDLib::Uniform uni(123456);

  std::vector<TwoDLib::FiducialElement> list;
  TwoDLib::TransitionMatrixGenerator gen(tree_bins,uni,1000,list);

  std::vector<TwoDLib::TransitionList> transitions;
  TwoDLib::TransitionList l;

  for( unsigned int i = 0; i < mesh.NrStrips(); i++ ){
    for (unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++ ){
      gen.Reset(1000);
      gen.GenerateTransformUsingQuadTranslation(i,j,tree,bins.allCoords());

      l._number = gen.N();
      l._origin = TwoDLib::Coordinates(i,j);
      l._destination_list = gen.HitList();

      std::cout << '\r' << std::setw(10) << 100.0 * ((float)(j+(i*mesh.NrCellsInStrip(1)))/(float)(mesh.NrStrips()*mesh.NrCellsInStrip(1))) << "%" << std::setfill(' ') << std::flush;

      transitions.push_back(l);
      
      ofst << "<cell>";
      	  ofst << "<coordinates>";
      	  	  ofst << i << "," << j;
      	  ofst << "</coordinates>";
      	  ofst << "<wbins>";
          std::map<unsigned int, double> props;
      	  for (auto h : l._destination_list){
            if (h._prop > 0.) {
              if (!props.count( h._cell[0] ))
                props[h._cell[0]] = h._prop;
              else
                props[h._cell[0]] += h._prop;
            }
      	  }
          for( std::map<unsigned int, double>::const_iterator it = props.begin(); it != props.end(); ++it )
            ofst << it->first << "," << it->second << ";";
      	  ofst << "</wbins>";
      	  ofst << "<vbins>";
      	  props.clear();
      	  for (auto h : l._destination_list){
            if (h._prop > 0.) {
              if (!props.count( h._cell[1] ))
                props[h._cell[1]] = h._prop;
              else
                props[h._cell[1]] += h._prop;
            }
      	  }
          for( std::map<unsigned int, double>::const_iterator it = props.begin(); it != props.end(); ++it )
            ofst << it->first << "," << it->second << ";";
      	  ofst << "</vbins>";
      ofst << "</cell>\n";
    }
  }
  	ofst << "</transitions>\n";
 }

void CalculateProjections
(
 std::ofstream& ofst,
 const TwoDLib::Mesh& mesh,
 double v_min,
 double v_max,
 unsigned int nv,
 double w_min,
 double w_max,
 unsigned int nw
 ){
  TwoDLib::Uniform uni(123456);
  ofst << "<transitions>\n";
  for( unsigned int i = 0; i < mesh.NrStrips(); i++ )
    for (unsigned int j = 0; j < mesh.NrCellsInStrip(i); j++ ){
      const TwoDLib::Cell& quad = mesh.Quad(i,j);
      TwoDLib::PolyGenerator gen(quad,uni);
      vector<TwoDLib::Point> vec_points(N_POINTS);
      gen.Generate(&vec_points);
      vector<float> vec_v(nv,0.), vec_w(nw,0.);
      for(const TwoDLib::Point& point: vec_points){
    	  unsigned int bin_v = Bin(point[0], v_min, v_max, nv);
    	  vec_v[bin_v]++;
    	  unsigned int bin_w = Bin(point[1], w_min, w_max, nw);
    	  vec_w[bin_w]++;
      }

      ofst << "<cell>";
      	  ofst << "<coordinates>";
      	  	  ofst << i << "," << j;
      	  ofst << "</coordinates>";
      	  ofst << "<vbins>";
      	  for (unsigned int i = 0; i < nv; i++){
      		  if (vec_v[i] > 0.)
      			  ofst << i << "," << vec_v[i]/N_POINTS << ";";
      	  }
      	  ofst << "</vbins>";
      	  ofst << "<wbins>";
      	  for (unsigned int i = 0; i < nw; i++){
      	  if (vec_w[i] > 0.)
      		  ofst << i << "," << vec_w[i]/N_POINTS << ";";
      	  }
      	  ofst << "</wbins>";
      ofst << "</cell>\n";
    }
  	ofst << "</transitions>\n";
}

void CreateProjections
(
 const string& projection_name,
 const TwoDLib::Mesh& mesh,
 double v_min,
 double v_max,
 unsigned int nv,
 double w_min,
 double w_max,
 unsigned int nw
 ){
  std::ofstream ofst(projection_name);
  ofst << "<Projection>\n";
  mesh.ToXML(ofst);

  ofst << "<V_limit>\n";
  ofst << "<V_min>";
  ofst << v_min;
  ofst << "</V_min>";
  ofst << "<V_max>";
  ofst << v_max;
  ofst << "</V_max>";
  ofst << "<N_V>";
  ofst << nv;
  ofst << "</N_V>\n";
  ofst << "</V_limit>\n";

  ofst << "<W_limit>\n";
  ofst << "<W_min>";
  ofst << w_min;
  ofst << "</W_min>";
  ofst << "<W_max>";
  ofst << w_max;
  ofst << "</W_max>";
  ofst << "<N_W>";
  ofst << nw;
  ofst << "</N_W>\n";
  ofst << "</W_limit>\n";

  CalculateProjectionsGeometric(ofst, mesh, v_min, v_max, nv, w_min, w_max, nw);

  std::cout << "\nComplete.\n";

  ofst << "</Projection>\n";
}


void ProduceProjectionFile
(
 const string& mesh_name,
 double v_min,
 double v_max,
 int nv,
 double w_min,
 double w_max,
 double nw
 ){
  // create the mesh
  TwoDLib::Mesh mesh(mesh_name);
  // some sanity checking
  std::pair<TwoDLib::Point, TwoDLib::Point> point_pair = Analyse(mesh);
  if ( point_pair.first[0]  < v_min ||
       point_pair.first[1]  < w_min ||
       point_pair.second[0] > v_max ||
       point_pair.second[1] > w_max )
    throw TwoDLib::TwoDLibException("Your binning doesn't cover the mesh");

  std::vector<string> elem;
  // Parse input arguments
  TwoDLib::split(mesh_name,'.',elem);

  string projection_name(elem[0] + ".projection");
  CreateProjections(projection_name, mesh, v_min, v_max, nv, w_min, w_max, nw);

}

int main(int argc, char** argv){

  try  {
    // There should be binning files and one mesh or model file.
    if (argc == 8){

      // Typical use, generate projection file
      std::string mesh_name(argv[1]);
      std::vector<string> elem;

      // Parse input arguments
      TwoDLib::split(mesh_name,'.',elem);

      if (elem.size() < 2 || elem[1] != string("model"))
	throw TwoDLib::TwoDLibException("Model extension not .model");

      std::istringstream ist_vmin(argv[2]);
      std::istringstream ist_vmax(argv[3]);
      std::istringstream ist_nv  (argv[4]);
      std::istringstream ist_wmin(argv[5]);
      std::istringstream ist_wmax(argv[6]);
      std::istringstream ist_nw  (argv[7]);

      double v_min, v_max, w_min, w_max;
      unsigned int nv, nw;
      ist_vmin >> v_min;
      ist_vmax >> v_max;
      ist_nv   >> nv;

      ist_wmin >> w_min;
      ist_wmax >> w_max;
      ist_nw   >> nw;

      ProduceProjectionFile(argv[1],v_min, v_max, nv, w_min, w_max, nw);

    } else if (argc == 2){
      // First determine size of the mesh
      std::cout << "Scanning model file" << std::endl;

      std::string mesh_name(argv[1]);
      std::vector<string> elem;

      // parse input arguments

      TwoDLib::split(mesh_name,'.',elem);

      if (elem.size() < 2 || elem[1] != string("model"))
	throw TwoDLib::TwoDLibException("Model extension not .model");

      // print some info about the mesh
      const TwoDLib::Mesh mesh(argv[1]);
      std::cout << "There are: " << mesh.NrStrips() << " strips in the mesh." << std::endl;

      // in particular, the bounding box
      std::pair<TwoDLib::Point,TwoDLib::Point> point_pair = Analyse(mesh);
      std::cout << "Bounding box: " << std::endl;
      std::cout << "Upper right: " << point_pair.second[0] << " " << point_pair.second[1] << std::endl;
      std::cout << "Lower left: "  << point_pair.first[0]  << " " << point_pair.first[1] << std::endl;
    } else {
      std::cout << "Usage: Projection <modelfile> <v_min>  <v_max> <n_points> <w_min> <w_max> <n_points> or" << std::endl;
      std::cout << "Usage: Projection <modelfile> to obtain grid boundaries" << std::endl;

    }
  }
  catch(const TwoDLib::TwoDLibException& e){
    std::cout << e.what() << std::endl;
  }

  return 0;
}
