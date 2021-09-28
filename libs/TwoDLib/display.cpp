#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <sstream>
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <boost/filesystem.hpp>

#include "display.hpp"
#include "include/glm/glm.hpp"
#include <MPILib/include/MPINetworkCode.hpp>
#include "Quadrilateral.hpp"

using namespace TwoDLib;
using namespace std::chrono;

Display* Display::disp = 0;

Display::Display(){
	lastTime = 0;
	write_frames = false;
	start_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
	_dws = std::map<MPILib::NodeId, DisplayWindow>();
}

Display::~Display(){
	if (glutGetWindow()){
		Display::getInstance()->updateDisplay(1);
		glutDestroyWindow(glutGetWindow());
	}
#ifndef USING_APPLE_GLUT // Hack to avoid issues with OSX glut version
	glutExit();
#endif
}

unsigned int Display::addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys, bool _3d) {
	return addOdeSystem(nid, sys, _3d, 0);
}

unsigned int Display::addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys, bool _3d, unsigned int mesh_index) {
	unsigned int index = _dws.size();

	DisplayWindow window;
	window._system = sys;
	window._mesh_index = mesh_index;
	window._3D = _3d;

	// Find extent of mesh to normalise to screen size

	Mesh m = sys->MeshObjects()[mesh_index];

	double mesh_min_v = 10000000.0;
	double mesh_max_v = -10000000.0;
	double mesh_min_h = 10000000.0;
	double mesh_max_h = -10000000.0;

	for(unsigned int i = 0; i<m.NrStrips(); i++){
		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
			Quadrilateral q = m.Quad(i,j);
			Point c = q.Centroid();
			if (c[0] > mesh_max_v)
				mesh_max_v = c[0];
			if (c[0] < mesh_min_v)
				mesh_min_v = c[0];
			if (c[1] > mesh_max_h)
				mesh_max_h = c[1];
			if (c[1] < mesh_min_h)
				mesh_min_h = c[1];
		}
	}

	// If there's only a single strip, then the min and max might be the same,
	// find the
	if (mesh_max_v == mesh_min_v){
		mesh_max_v = 0.005;
		mesh_min_v = 0.0;
	}

	if (mesh_min_h == mesh_max_h){
		mesh_max_h = 0.005;
		mesh_min_h = 0.0;
	}

	window.mesh_min_v = mesh_min_v;
	window.mesh_max_v = mesh_max_v;
	window.mesh_min_h = mesh_min_h;
	window.mesh_max_h = mesh_max_h;

	std::map<MPILib::NodeId,DisplayWindow>::iterator it = _dws.find(nid);
	if(it == _dws.end())
		_dws.insert(std::make_pair(nid, window));

	return index;
}

// The OpenGL display function, called as fast as possible.
void Display::display(void) {
	if (_dws.size() == 0)
		return;

	milliseconds real_time = duration_cast< milliseconds >(
    system_clock::now().time_since_epoch());
	milliseconds time_elapsed = real_time - start_time;

	int window_index = 0;
	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter){
		if (iter->second._window_index == glutGetWindow())
			window_index = iter->first;
	}

	// if (time_elapsed.count() % 10 != 0)
	// 	return;

	glClearColor(0.0f, 0.0f, 0.2f, 0.0f);

	glClear(GL_COLOR_BUFFER_BIT);

	// **** used for 3D ****
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glPushMatrix();
	// glTranslatef(0, 0, -1);

	glBegin(GL_QUADS);

	Mesh m = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index];

	double max = -99999999.0;
	for(unsigned int i = 0; i<m.NrStrips(); i++){
		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
			double cell_area = std::abs(m.Quad(i,j).SignedArea());
			if(_dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j)]/cell_area == 0){
				continue;
			}
			if (max < log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j)]/cell_area))
				max = log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j)]/cell_area);
		}
	}

	double min = 999999.0;
	for(unsigned int i = 0; i<m.NrStrips(); i++){
		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
			double cell_area = std::abs(m.Quad(i,j).SignedArea());
			if(_dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j)]/cell_area == 0){
				continue;
			}

			if (log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j)]/cell_area) < min)
				min = log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j)]/cell_area);
		}
	}

	double mesh_min_v = _dws[window_index].mesh_min_v;
	double mesh_max_v = _dws[window_index].mesh_max_v;
	double mesh_min_h = _dws[window_index].mesh_min_h;
	double mesh_max_h = _dws[window_index].mesh_max_h;

	for(unsigned int i = 0; i<m.NrStrips(); i++){
		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
			unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j);
			Quadrilateral q = m.Quad(i,j);
			double cell_area = std::abs(q.SignedArea());
			double mass = 0.0;
			if (_dws[window_index]._system->Mass()[idx] / cell_area != 0 && _dws[window_index]._system->FiniteSizeNumObjects()[_dws[window_index]._mesh_index] == 0) {
				mass = std::min(1.0, std::max(0.0, (log10(_dws[window_index]._system->Mass()[idx] / cell_area) - min) / (max - min)));
			}
			else {
				if (_dws[window_index]._system->_vec_cells_to_objects[idx].size() > 0)
					//mass = (double)_dws[window_index]._system->_vec_cells_to_objects[idx].size() / (double)_dws[window_index]._system->_num_objects;
					mass = 1.0;
			}

			vector<Point> ps = q.Points();

			glColor3f(std::min(1.0,mass*2.0), std::max(0.0,((mass*2.0) - 1.0)), 0);
			glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
		}
	}

	glEnd();

	// Print real time and sim time

	double sim_time = 0.0;

	sim_time = _current_sim_it * _time_step;

	glColor3f( 1.0, 1.0, 1.0 );
  glRasterPos2f(0.3, 0.9);
  int len, i;
	std::string t = std::string("Sim Time (s) : ") + std::to_string( sim_time );
	const char* c_string = t.c_str();
  len = (int)strlen( c_string );
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
  }

	double h_width = (mesh_max_h - mesh_min_h);
	char buff[32];
  sprintf(buff, "%.*g", 1, h_width);
	double h_step = (double)std::atof(buff) / 10.0;

	std::string s_h_step = std::to_string(h_step);
	s_h_step.pop_back();
	h_step = std::stod(s_h_step);
	double nice_min_h = (double)floor(mesh_min_h/h_step) * h_step;
	double nice_max_h = (double)ceil(mesh_max_h/h_step) * h_step;

	double v_width = (mesh_max_v - mesh_min_v);
  sprintf(buff, "%.*g", 1, v_width);
	double v_step = (double)std::atof(buff) / 10.0;

	std::string s_v_step = std::to_string(v_step);
	s_v_step.pop_back();
	v_step = std::stod(s_v_step);
	double nice_min_v = (double)floor(mesh_min_v/v_step) * v_step;
	double nice_max_v = (double)ceil(mesh_max_v/v_step) * v_step;

	double pos = nice_min_h;
	while(pos < nice_max_h){
		if (std::abs(pos) < 0.0000000001 )
			pos = 0.0;

		glColor3f( 1.0, 1.0, 1.0 );
	  glRasterPos2f(-1.0, 2*((pos - (mesh_min_h + ((mesh_max_h - mesh_min_h)/2.0)))/(mesh_max_h - mesh_min_h)));

		std::stringstream stream;
		stream <<  std::setprecision(3) << pos;
		t = stream.str();
		c_string = t.c_str();
	  len = (int)strlen( c_string );
	  for (i = 0; i < len; i++) {
	    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
	  }
		pos += h_step;
	}

	pos = nice_min_v;
	while(pos < nice_max_v){
		if (std::abs(pos) < 0.0000000001 )
			pos = 0.0;
		glRasterPos2f(2*((pos - (mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v)), -1.0);
		std::stringstream stream2;
		stream2 <<  std::setprecision(3) << pos;
		t = stream2.str();
		c_string = t.c_str();
		len = (int)strlen( c_string );
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
		}
		pos += (nice_max_v - nice_min_v) / 10;
	}

	// **** used for 3D ****
	// glPopMatrix();
	glutSwapBuffers();
	glFlush();

	if(write_frames)
		writeFrame(window_index,_current_sim_it);
}

std::vector<double> mat_mult(std::vector<std::vector<double>>& mat, std::vector<double>& vector) {
	std::vector<double> np = { 0.0,0.0,0.0,0.0 };
	for (unsigned int i = 0; i < 4; i++) {
		for (unsigned int j = 0; j < 4; j++) {
			np[i] += vector[j] * mat[i][j];
		}
	}
	return np;
}

// The OpenGL display function, called as fast as possible.
void Display::display_3d(void) {
	if (_dws.size() == 0)
		return;

	static unsigned int num_frames = 0;
	bool log_scale = true;

	// lets go a bit faster in our rendering eh?
	// if(num_frames++ % 2 != 0)
	// 	return;

	static float rot = 0.0;

	milliseconds real_time = duration_cast<milliseconds>(
		system_clock::now().time_since_epoch());
	milliseconds time_elapsed = real_time - start_time;

	int window_index = 0;
	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter) {
		if (iter->second._window_index == glutGetWindow())
			window_index = iter->first;
	}

	if (time_elapsed.count() % 10 != 0)
	 	return;

	glClearColor(0.0f, 0.0f, 0.2f, 1.0f);

	//glClear(GL_COLOR_BUFFER_BIT);

	// **** used for 3D ****
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPushMatrix();
	glTranslatef(0, 0, -1);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glBegin(GL_QUADS);

	Mesh m = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index];

	double max = -99999999.0;
	for (unsigned int i = 0; i < m.NrStrips(); i++) {
		for (unsigned int j = 0; j < m.NrCellsInStrip(i); j++) {
			double cell_area = std::abs(m.Quad(i, j).SignedArea());
			if (_dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area == 0) {
				continue;
			}
			if (!log_scale) {
				if (max < 1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area)
					max = 1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area;
			}
			else {
				if (max < log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area))
					max = log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area);
			}

		}
	}

	double min = 99999999999.0;
	for (unsigned int i = 0; i < m.NrStrips(); i++) {
		for (unsigned int j = 0; j < m.NrCellsInStrip(i); j++) {
			double cell_area = std::abs(m.Quad(i, j).SignedArea());
			if (_dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area == 0) {
				continue;
			}

			if (!log_scale)
			{
				if (1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area < min)
					min = 1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area;
			}
			else {
				if (log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area) < min)
					min = log10(1e-6 + _dws[window_index]._system->Mass()[_dws[window_index]._system->Map(_dws[window_index]._mesh_index, i, j)] / cell_area);
			}

		}
	}

	double mesh_min_v = _dws[window_index].mesh_min_v;
	double mesh_max_v = _dws[window_index].mesh_max_v;
	double mesh_min_h = _dws[window_index].mesh_min_h;
	double mesh_max_h = _dws[window_index].mesh_max_h;

	// display for individual

	// for(unsigned int idx : _dws[window_index]._system->_individuals){
	// 	Coordinates c = _dws[window_index]._system->toCoords(idx);
	// 	Quadrilateral q = m.Quad(c[0],c[1]);
	// 	vector<Point> ps = q.Points();

	// 	glColor3f(1.0, 0, 0);
	// 	glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 	glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 	glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 	glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// }

	// Display 2D mass

	// for(unsigned int i = 0; i<m.NrStrips(); i++){
	// 	for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
	// 		unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j);
	// 		Cell q = m.Quad(i,j);
	// 		double cell_area = std::abs(q.SignedArea());
	// 		double mass = 0.0;
	// 		if (_dws[window_index]._system->Mass()[idx]/cell_area != 0)
	// 			mass = std::min(1.0,std::max(0.0,(log10(_dws[window_index]._system->Mass()[idx]/cell_area) - min) / (max-min)));
	// 		vector<Point> ps = q.Points();

	// 		glColor3f(std::min(1.0,mass*2.0), std::max(0.0,((mass*2.0) - 1.0)), 0);
	// 		glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 	}
	// }

	// Display 2D layered mass

	// for(unsigned int i = 0; i<100; i++){
	// 	for(unsigned int k = 0; k<100; k++){
	// 		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
	// 			unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index,(k*100)+i,j);
	// 			Cell q = m.Quad(i,j);
	// 			double cell_area = std::abs(q.SignedArea());
	// 			double mass = 0.0;
	// 			if (_dws[window_index]._system->Mass()[idx]/cell_area != 0)
	// 				mass = std::min(1.0,std::max(0.0,(log10(_dws[window_index]._system->Mass()[idx]/cell_area) - min) / (max-min)));
	// 			vector<Point> ps = q.Points();

	// 			// glColor4f(std::min(1.0,mass*2.0), std::max(0.0,((mass*2.0) - 1.0)), 0, std::min(1.0,mass*2.0));
	// 			glColor4f(1.0, 0.0, 0, std::min(1.0,mass*2.0));
	// 			// glColor4f(1.0, 0.0, 0, 0.01);
	// 			glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 			glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 			glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 			glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		}
	// 	}
	// }

	// Display 2D layered flattened

	// for(unsigned int i = 0; i<100; i++){
	// 	for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {

	// 		unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j);
	// 		Cell q = m.Quad(i,j);
	// 		vector<Point> ps = q.Points();

	// 		double mass = 0.0;
	// 		double cell_area = std::abs(q.SignedArea());
	// 		for(unsigned int k = 0; k<100; k++){
	// 			unsigned int idxx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index,(k*100)+i,j);
	// 			if (mass < std::min(1.0,std::max(0.0,(log10(_dws[window_index]._system->Mass()[idxx]/cell_area) - min) / (max-min)))){
	// 				if (_dws[window_index]._system->Mass()[idx]/cell_area != 0)
	// 					mass += std::min(1.0,std::max(0.0,(log10(_dws[window_index]._system->Mass()[idxx]/cell_area) - min) / (max-min)));
	// 			}			
	// 		}

	// 		// glColor4f(std::min(1.0,mass*2.0), std::max(0.0,((mass*2.0) - 1.0)), 0, std::min(1.0,mass*2.0));
	// 		glColor4f(1.0, 0.0, 0, std::min(1.0,mass*2.0));
	// 		// glColor4f(1.0, 0.0, 0, 0.01);
	// 		glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 		glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
	// 	}
	// }

	// Display 3D mass

	double rot_y_angle = -M_PI_4 + 3.6 + (rot);
	double rot_x_angle = (-M_PI_4 / 2.0);

	// rot += 0.01;

	double x_scale = 1.0;
	double y_scale = 1.0;
	double z_scale = 1.0;

	double x_pos = 0.0;
	double y_pos = 0.0;
	double z_pos = -1.0;

	unsigned int size_x = 100;
	unsigned int size_y = 30;
	unsigned int size_z = 30;

	std::vector<std::vector<double>> world_scale = { {x_scale,0.0,0.0,0.0},{0.0,y_scale,0.0,0.0},{0.0,0.0,z_scale,0.0},{0.0,0.0,0.0,1.0} };
	std::vector<std::vector<double>> world_x_rotation = { {1.0,0.0,0.0,0.0},{0.0,cos(rot_x_angle),sin(rot_x_angle),0.0},{0.0,-sin(rot_x_angle),cos(rot_x_angle),0.0},{0.0,0.0,0.0,1.0} };
	std::vector<std::vector<double>> world_y_rotation = { {cos(rot_y_angle),0.0,sin(rot_y_angle),0.0},{0.0,1.0,0.0,0.0},{-sin(rot_y_angle),0.0,cos(rot_y_angle),0.0},{0.0,0.0,0.0,1.0} };
	std::vector<std::vector<double>> world_translate = { {1.0,0.0,0.0,x_pos},{0.0,1.0,0.0,y_pos},{0.0,0.0,1.0,z_pos},{0.0,0.0,0.0,1.0} };

	unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index, 0, 12);
	std::cout << _dws[window_index]._system->Mass()[idx] << "\n";

	double max_mass = 0.0;
	for (unsigned int i = 0; i < size_z; i++) {
		for (unsigned int k = 0; k < size_y; k++) {
			for (unsigned int j = 0; j < m.NrCellsInStrip(i); j++) {
				unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index, (i * size_y) + k, j);

				if (i != 0 && k != 0 && j != 0 && i != size_z - 1 && k != size_y - 1 && j != size_x - 1)
					if (_dws[window_index]._system->Mass()[idx] == 0) continue; // skip if mass is basically nothing

				double cell_area = std::abs(m.Quad(0, 0).SignedArea());
				double mass = 0.0;
				if (!log_scale) {
					if (cell_area != 0 && _dws[window_index]._system->Mass()[idx] > 0.0)
						mass = (_dws[window_index]._system->Mass()[idx] / cell_area - min) / (max - min);
				}
				else {
					if (cell_area != 0 && _dws[window_index]._system->Mass()[idx] > 0.0)
						mass = (log10(_dws[window_index]._system->Mass()[idx] / cell_area) - min) / (max - min);
				}

				if (i == 0 || k == 0 || j == 0 || i == size_z - 1 || k == size_y - 1 || j == size_x - 1) {
					glColor4f(1.0, 1.0, 1.0, 0.01);
				}

				if (mass > 0.00000001)
					glColor4f(std::min(1.0, mass * 2.0), std::max(0.0, ((mass * 2.0) - 1.0)), 0, mass);

				double cell_x = -0.5 + j * (1.0 / size_x);
				double cell_y = -0.5 + i * (1.0 / size_z);
				double cell_z = -0.5 + k * (1.0 / size_y);

				double half_cell_x_width = 0.95 * (0.5 / size_x);
				double half_cell_y_width = 0.95 * (0.5 / size_z);
				double half_cell_z_width = 0.95 * (0.5 / size_y);

				std::vector<double> p1 = { cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
				std::vector<double> p2 = { cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
				std::vector<double> p3 = { cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
				std::vector<double> p4 = { cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
				std::vector<double> p5 = { cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
				std::vector<double> p6 = { cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
				std::vector<double> p7 = { cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
				std::vector<double> p8 = { cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0 };

				p1 = mat_mult(world_scale, p1);
				p2 = mat_mult(world_scale, p2);
				p3 = mat_mult(world_scale, p3);
				p4 = mat_mult(world_scale, p4);
				p5 = mat_mult(world_scale, p5);
				p6 = mat_mult(world_scale, p6);
				p7 = mat_mult(world_scale, p7);
				p8 = mat_mult(world_scale, p8);

				p1 = mat_mult(world_y_rotation, p1);
				p2 = mat_mult(world_y_rotation, p2);
				p3 = mat_mult(world_y_rotation, p3);
				p4 = mat_mult(world_y_rotation, p4);
				p5 = mat_mult(world_y_rotation, p5);
				p6 = mat_mult(world_y_rotation, p6);
				p7 = mat_mult(world_y_rotation, p7);
				p8 = mat_mult(world_y_rotation, p8);

				p1 = mat_mult(world_x_rotation, p1);
				p2 = mat_mult(world_x_rotation, p2);
				p3 = mat_mult(world_x_rotation, p3);
				p4 = mat_mult(world_x_rotation, p4);
				p5 = mat_mult(world_x_rotation, p5);
				p6 = mat_mult(world_x_rotation, p6);
				p7 = mat_mult(world_x_rotation, p7);
				p8 = mat_mult(world_x_rotation, p8);

				p1 = mat_mult(world_translate, p1);
				p2 = mat_mult(world_translate, p2);
				p3 = mat_mult(world_translate, p3);
				p4 = mat_mult(world_translate, p4);
				p5 = mat_mult(world_translate, p5);
				p6 = mat_mult(world_translate, p6);
				p7 = mat_mult(world_translate, p7);
				p8 = mat_mult(world_translate, p8);

				// front face bad
				glVertex3f(p1[0], p1[1], p1[2]);
				glVertex3f(p2[0], p2[1], p2[2]);
				glVertex3f(p3[0], p3[1], p3[2]);
				glVertex3f(p4[0], p4[1], p4[2]);

				// back face bad
				glVertex3f(p5[0], p5[1], p5[2]);
				glVertex3f(p6[0], p6[1], p6[2]);
				glVertex3f(p7[0], p7[1], p7[2]);
				glVertex3f(p8[0], p8[1], p8[2]);

				// right face
				glVertex3f(p4[0], p4[1], p4[2]);
				glVertex3f(p3[0], p3[1], p3[2]);
				glVertex3f(p7[0], p7[1], p7[2]);
				glVertex3f(p8[0], p8[1], p8[2]);

				// left face
				glVertex3f(p1[0], p1[1], p1[2]);
				glVertex3f(p2[0], p2[1], p2[2]);
				glVertex3f(p6[0], p6[1], p6[2]);
				glVertex3f(p5[0], p5[1], p5[2]);

				// top face good
				glVertex3f(p2[0], p2[1], p2[2]);
				glVertex3f(p6[0], p6[1], p6[2]);
				glVertex3f(p7[0], p7[1], p7[2]);
				glVertex3f(p3[0], p3[1], p3[2]);

				// bottom face
				glVertex3f(p1[0], p1[1], p1[2]);
				glVertex3f(p5[0], p5[1], p5[2]);
				glVertex3f(p8[0], p8[1], p8[2]);
				glVertex3f(p4[0], p4[1], p4[2]);

			}
		}
	}

	//3D individuals

	// for(unsigned int idx : _dws[window_index]._system->_individuals){

	// 	double cell_area = std::abs(m.Quad(0,0).SignedArea());
	// 	double mass = 0.0;
	// 	if (cell_area != 0 && _dws[window_index]._system->Mass()[idx] > 0.0)
	// 		mass = (log10(_dws[window_index]._system->Mass()[idx]/cell_area) - min) / (max-min);

	// 	glColor4f(std::min(1.0,mass*2.0), std::max(0.0,((mass*2.0) - 1.0)), 0, mass);

	// 	unsigned int j = (idx % 200);
	// 	unsigned int i = (idx - (idx % 200)) / (200);
	// 	unsigned int k = (idx - (idx % 200)) / (200*200);

	// 	double cell_x = -0.5 + j*(1.0/size_x);
	// 	double cell_y = -0.5 + i*(1.0/size_z);
	// 	double cell_z = -0.5 + k*(1.0/size_y);

	// 	double half_cell_x_width = 0.95*(0.5/size_x);
	// 	double half_cell_y_width = 0.95*(0.5/size_z);
	// 	double half_cell_z_width = 0.95*(0.5/size_y);

	// 	std::vector<double> p1 = {cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0};
	// 	std::vector<double> p2 = {cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0};
	// 	std::vector<double> p3 = {cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0};
	// 	std::vector<double> p4 = {cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0};
	// 	std::vector<double> p5 = {cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0};
	// 	std::vector<double> p6 = {cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0};
	// 	std::vector<double> p7 = {cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0};
	// 	std::vector<double> p8 = {cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0};

	// 	p1 = mat_mult(world_scale, p1);
	// 	p2 = mat_mult(world_scale, p2);
	// 	p3 = mat_mult(world_scale, p3);
	// 	p4 = mat_mult(world_scale, p4);
	// 	p5 = mat_mult(world_scale, p5);
	// 	p6 = mat_mult(world_scale, p6);
	// 	p7 = mat_mult(world_scale, p7);
	// 	p8 = mat_mult(world_scale, p8);

	// 	p1 = mat_mult(world_y_rotation, p1);
	// 	p2 = mat_mult(world_y_rotation, p2);
	// 	p3 = mat_mult(world_y_rotation, p3);
	// 	p4 = mat_mult(world_y_rotation, p4);
	// 	p5 = mat_mult(world_y_rotation, p5);
	// 	p6 = mat_mult(world_y_rotation, p6);
	// 	p7 = mat_mult(world_y_rotation, p7);
	// 	p8 = mat_mult(world_y_rotation, p8);

	// 	p1 = mat_mult(world_x_rotation, p1);
	// 	p2 = mat_mult(world_x_rotation, p2);
	// 	p3 = mat_mult(world_x_rotation, p3);
	// 	p4 = mat_mult(world_x_rotation, p4);
	// 	p5 = mat_mult(world_x_rotation, p5);
	// 	p6 = mat_mult(world_x_rotation, p6);
	// 	p7 = mat_mult(world_x_rotation, p7);
	// 	p8 = mat_mult(world_x_rotation, p8);

	// 	p1 = mat_mult(world_translate, p1);
	// 	p2 = mat_mult(world_translate, p2);
	// 	p3 = mat_mult(world_translate, p3);
	// 	p4 = mat_mult(world_translate, p4);
	// 	p5 = mat_mult(world_translate, p5);
	// 	p6 = mat_mult(world_translate, p6);
	// 	p7 = mat_mult(world_translate, p7);
	// 	p8 = mat_mult(world_translate, p8);

	// 	// front face bad
	// 	glVertex3f(p1[0], p1[1], p1[2]);
	// 	glVertex3f(p2[0], p2[1], p2[2]);
	// 	glVertex3f(p3[0], p3[1], p3[2]);
	// 	glVertex3f(p4[0], p4[1], p4[2]);

	// 	// back face bad
	// 	glVertex3f(p5[0], p5[1], p5[2]);
	// 	glVertex3f(p6[0], p6[1], p6[2]);
	// 	glVertex3f(p7[0], p7[1], p7[2]);
	// 	glVertex3f(p8[0], p8[1], p8[2]);

	// 	// right face
	// 	glVertex3f(p4[0], p4[1], p4[2]);
	// 	glVertex3f(p3[0], p3[1], p3[2]);
	// 	glVertex3f(p7[0], p7[1], p7[2]);
	// 	glVertex3f(p8[0], p8[1], p8[2]);

	// 	// left face
	// 	glVertex3f(p1[0], p1[1], p1[2]);
	// 	glVertex3f(p2[0], p2[1], p2[2]);
	// 	glVertex3f(p6[0], p6[1], p6[2]);
	// 	glVertex3f(p5[0], p5[1], p5[2]);

	// 	// top face good
	// 	glVertex3f(p2[0], p2[1], p2[2]);
	// 	glVertex3f(p6[0], p6[1], p6[2]);
	// 	glVertex3f(p7[0], p7[1], p7[2]);
	// 	glVertex3f(p3[0], p3[1], p3[2]);

	// 	// bottom face
	// 	glVertex3f(p1[0], p1[1], p1[2]);
	// 	glVertex3f(p5[0], p5[1], p5[2]);
	// 	glVertex3f(p8[0], p8[1], p8[2]);
	// 	glVertex3f(p4[0], p4[1], p4[2]);

	// }

	glEnd();

	// Print real time and sim time

	double sim_time = 0.0;

	sim_time = _current_sim_it * _time_step;

	glColor3f(1.0, 1.0, 1.0);
	glRasterPos2f(0.3, 0.9);
	int len, i;
	std::string t = std::string("Sim Time (s) : ") + std::to_string(sim_time);
	const char* c_string = t.c_str();
	len = (int)strlen(c_string);
	for (i = 0; i < len; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
	}

	double h_width = (mesh_max_h - mesh_min_h);
	char buff[32];
	sprintf(buff, "%.*g", 1, h_width);
	double h_step = (double)std::atof(buff) / 10.0;

	std::string s_h_step = std::to_string(h_step);
	s_h_step.pop_back();
	h_step = std::stod(s_h_step);
	double nice_min_h = (double)floor(mesh_min_h / h_step) * h_step;
	double nice_max_h = (double)ceil(mesh_max_h / h_step) * h_step;

	double v_width = (mesh_max_v - mesh_min_v);
	sprintf(buff, "%.*g", 1, v_width);
	double v_step = (double)std::atof(buff) / 10.0;

	std::string s_v_step = std::to_string(v_step);
	s_v_step.pop_back();
	v_step = std::stod(s_v_step);
	double nice_min_v = (double)floor(mesh_min_v / v_step) * v_step;
	double nice_max_v = (double)ceil(mesh_max_v / v_step) * v_step;

	double pos = nice_min_h;
	double scaled_pos = nice_min_h;
	while (pos < nice_max_h) {
		if (std::abs(pos) < 0.0000000001) {
			pos = 0.0;
			scaled_pos = 0.0;
		}

		glColor3f(1.0, 1.0, 1.0);
		glRasterPos2f(-1.0, 2 * ((pos - (mesh_min_h + ((mesh_max_h - mesh_min_h) / 2.0))) / (mesh_max_h - mesh_min_h)));

		std::stringstream stream;
		stream << std::setprecision(3) << scaled_pos;
		t = stream.str();
		c_string = t.c_str();
		len = (int)strlen(c_string);
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
		}
		pos += h_step;
		scaled_pos += h_step * 100.0;
	}

	pos = nice_min_v;
	while (pos < nice_max_v) {
		if (std::abs(pos) < 0.0000000001)
			pos = 0.0;
		glRasterPos2f(2 * ((pos - (mesh_min_v + ((mesh_max_v - mesh_min_v) / 2.0))) / (mesh_max_v - mesh_min_v)), -1.0);
		std::stringstream stream2;
		stream2 << std::setprecision(3) << pos;
		t = stream2.str();
		c_string = t.c_str();
		len = (int)strlen(c_string);
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
		}
		pos += (nice_max_v - nice_min_v) / 10;
	}

	// **** used for 3D ****
	glPopMatrix();
	glutSwapBuffers();
	glFlush();

	if (write_frames)
		writeFrame(window_index, _current_sim_it);
}

void Display::writeFrame(unsigned int system, long frame_num){
	//This prevents the images getting padded
 	// when the width multiplied by 3 is not a multiple of 4
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  // width * height * 3
  int nSize = 500*500*3;
  // First let's create our buffer, 3 channels per Pixel
  char* dataBuffer = (char*)malloc(nSize*sizeof(char));

  if (!dataBuffer) return;

   // Let's fetch them from the backbuffer
   // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
  //  glReadPixels((GLint)0, (GLint)0,
	// 	(GLint)w, (GLint)h,
	// 	 GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);
	 glReadPixels((GLint)0, (GLint)0,
	 (GLint)500, (GLint)500,
		GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);

		const std::string dirname = std::string("node_") + std::to_string(system);

		if (! boost::filesystem::exists(dirname) ){
			boost::filesystem::create_directory(dirname);
		}

   //Now the file creation
	 std::string filename =  dirname + std::string("/") + std::to_string(frame_num) + std::string(".tga");
   FILE *filePtr = fopen(filename.c_str(), "wb");
   if (!filePtr) return;

   unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
  //  unsigned char header[6] = { w%256,w/256,
	// 		       h%256,h/256,
	// 		       24,0};
	 unsigned char header[6] = { 500%256,500/256,
 						500%256,500/256,
 						24,0};
   // We write the headers
   fwrite(TGAheader,	sizeof(unsigned char),	12,	filePtr);
   fwrite(header,	sizeof(unsigned char),	6,	filePtr);
   // And finally our image data
   fwrite(dataBuffer,	sizeof(GLubyte),	nSize,	filePtr);
   fclose(filePtr);

   free(dataBuffer);
}

void Display::scene(int width, int height)
{
	glViewport(0, 0, width, height);
	glLoadIdentity();
}

void Display::scene_3d(int width, int height)
{
	if (height == 0)
	{
		height = 1;
	}

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
 
	gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
 
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);
	glLoadIdentity();
}

void Display::init() const {
	gluPerspective(45, 1, 2, 10);
	glEnable(GL_DEPTH_TEST);
	//glClearColor(0.0f, 0.0f, 0.2f, 0.0f);
}

void Display::init_3d() const {
	gluPerspective(45, 1, 2, 10);
	glEnable(GL_DEPTH_TEST);
	//glClearColor(0.0f, 0.0f, 0.2f, 0.0f);
}

void Display::update() {
}

void Display::update_3d() {
}

void Display::updateDisplay(long current_sim_it) {
	int time;
	time = glutGet(GLUT_ELAPSED_TIME);
	Display::getInstance()->_current_sim_it = current_sim_it;
	lastTime = time;
	//Sleep(50);
	for (MPILib::NodeId id = 0; id < _nodes_to_display.size(); id++) {
		if(!glutGetWindow())
			continue;
		glutSetWindow(_dws[_nodes_to_display[id]]._window_index);
		glutPostRedisplay();
	}
#ifndef USING_APPLE_GLUT
	glutMainLoopEvent();
#else
	glutCheckLoop();
#endif

}

void Display::shutdown() const {
#ifndef USING_APPLE_GLUT // Hack to avoid issues with OSX glut version
	glutExit();
#endif

	// Nice new line if we quit early.
	std::cout << "\n";
}

void Display::animate(bool _write_frames, std::vector<MPILib::NodeId> nodes_to_display, double time_step) const{

	Display::getInstance()->_nodes_to_display = nodes_to_display;
	Display::getInstance()->write_frames = _write_frames;
	Display::getInstance()->_time_step = time_step;

	char* arv[] = {"Miind"};
	int count = 1;
	glutInit(&count, arv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);

	for (MPILib::NodeId id = 0; id < Display::getInstance()->_nodes_to_display.size(); id++) {
		if (!Display::getInstance()->_dws[Display::getInstance()->_nodes_to_display[id]]._3D) {
			Display::getInstance()->_dws[Display::getInstance()->_nodes_to_display[id]]._window_index = glutCreateWindow("Miind2D");
			glutDisplayFunc(Display::stat_display);
			glutReshapeFunc(Display::stat_scene);
			glutIdleFunc(Display::stat_update);
		}
		else {
			Display::getInstance()->_dws[Display::getInstance()->_nodes_to_display[id]]._window_index = glutCreateWindow("Miind3D");
			glutDisplayFunc(Display::stat_display_3d);
			glutReshapeFunc(Display::stat_scene_3d);
			glutIdleFunc(Display::stat_update_3d);
		}
		
	}

	atexit(Display::stat_shutdown);
// glutSetOption is not available in OSX glut - on other OSs (using freeglut), this allows us to keep running the simulation 
// even though the window is closed
// I don't know what will happen on OSX because I don't live and work in Shoreditch. 
#ifndef USING_APPLE_GLUT
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
#endif
	init();
}

void Display::processDraw(void) {
}
