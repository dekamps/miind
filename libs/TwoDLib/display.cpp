#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <thread>
#include <sstream>
#include <numeric>
#include <algorithm>
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <boost/filesystem.hpp>

#include "display.hpp"
#include "include/glm/glm.hpp"
#include <MPILib/include/MPINetworkCode.hpp>
#include "Quadrilateral.hpp"

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768
#define SHOW_TEXT true
#define SHOW_3D_SPACE true

using namespace TwoDLib;
using namespace std::chrono;

Display* Display::disp = 0;

Display::Display(){
	lastTime = 0;
	write_frames = false;
	start_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
	_dws = std::map<MPILib::NodeId, DisplayWindow>();
	upPressed = false;
	downPressed = false;
	leftPressed = false;
	rightPressed = false;
	pgdnPressed = false;
	pgupPressed = false;
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
	window.rot_x = 0.0;
	window.rot_y = 0.0;
	window.dim_select = 0;
	window.max_mass = -9999999;
	window.min_mass = 9999999;

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

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glClear(GL_COLOR_BUFFER_BIT);

	// **** used for 3D ****
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glPushMatrix();
	// glTranslatef(0, 0, -1);

	glBegin(GL_QUADS);

	std::vector<double> inferno_r_range(8);
	std::vector<double> inferno_g_range(8);
	std::vector<double> inferno_b_range(8);
	std::vector<double> inferno_values(8);

	for (unsigned int i = 0; i < 8; i++) {
		inferno_values[i] = (1.0 / (double)8) * (double)i;
	}

	inferno_r_range[0] = (double)0 / (double)255;
	inferno_g_range[0] = (double)0 / (double)255;
	inferno_b_range[0] = (double)4 / (double)255;

	inferno_r_range[1] = (double)40 / (double)255;
	inferno_g_range[1] = (double)11 / (double)255;
	inferno_b_range[1] = (double)84 / (double)255;

	inferno_r_range[2] = (double)101 / (double)255;
	inferno_g_range[2] = (double)21 / (double)255;
	inferno_b_range[2] = (double)110 / (double)255;

	inferno_r_range[3] = (double)159 / (double)255;
	inferno_g_range[3] = (double)42 / (double)255;
	inferno_b_range[3] = (double)99 / (double)255;

	inferno_r_range[4] = (double)212 / (double)255;
	inferno_g_range[4] = (double)72 / (double)255;
	inferno_b_range[4] = (double)66 / (double)255;

	inferno_r_range[5] = (double)245 / (double)255;
	inferno_g_range[5] = (double)125 / (double)255;
	inferno_b_range[5] = (double)21 / (double)255;

	inferno_r_range[6] = (double)250 / (double)255;
	inferno_g_range[6] = (double)193 / (double)255;
	inferno_b_range[6] = (double)39 / (double)255;

	inferno_r_range[7] = (double)252 / (double)255;
	inferno_g_range[7] = (double)255 / (double)255;
	inferno_b_range[7] = (double)164 / (double)255;

	Mesh m = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index];

	double mesh_min_v = _dws[window_index].mesh_min_v;
	double mesh_max_v = _dws[window_index].mesh_max_v;
	double mesh_min_h = _dws[window_index].mesh_min_h;
	double mesh_max_h = _dws[window_index].mesh_max_h;

	double max_m = -9999999;
	double min_m = 9999999;
	for(unsigned int i = 0; i<m.NrStrips(); i++){
		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {

			unsigned int idx = _dws[window_index]._system->Map(_dws[window_index]._mesh_index,i,j);
			
			Quadrilateral q = m.Quad(i,j);
			double cell_area = std::abs(q.SignedArea());
			double mass = 0.0;

			if (max_m < log10(1e-6 + _dws[window_index]._system->Mass()[idx] / cell_area))
				max_m = log10(1e-6 + _dws[window_index]._system->Mass()[idx] / cell_area);
			if (min_m >= log10(1e-6 + _dws[window_index]._system->Mass()[idx] / cell_area))
				min_m = log10(1e-6 + _dws[window_index]._system->Mass()[idx] / cell_area);

			/*if (max_m < _dws[window_index]._system->Mass()[idx])
				max_m = _dws[window_index]._system->Mass()[idx];
			if (min_m >= _dws[window_index]._system->Mass()[idx])
				min_m = _dws[window_index]._system->Mass()[idx];*/

			if (_dws[window_index]._system->Mass()[idx] / cell_area != 0 && _dws[window_index]._system->FiniteSizeNumObjects()[_dws[window_index]._mesh_index] == 0) {
				mass = std::min(1.0, std::max(0.0, (log10(1e-6 + _dws[window_index]._system->Mass()[idx] / cell_area) - _dws[window_index].min_mass) / (_dws[window_index].max_mass - _dws[window_index].min_mass)));
				//mass = std::min(1.0, std::max(0.0, (_dws[window_index]._system->Mass()[idx] - _dws[window_index].min_mass) / (_dws[window_index].max_mass - _dws[window_index].min_mass)));
			}
			else if (_dws[window_index]._system->FiniteSizeNumObjects()[_dws[window_index]._mesh_index] > 0){
				if (_dws[window_index]._system->_vec_cells_to_objects[idx].size() > 0)
				    mass = 0.5 + 1000.0 * ((double)_dws[window_index]._system->_vec_cells_to_objects[idx].size() / (double)_dws[window_index]._system->FiniteSizeNumObjects()[_dws[window_index]._mesh_index]);
					//mass = 1.0;
			}
			vector<Point> ps = q.Points();

			unsigned int lower_num = int(mass * 8.0);
			if (lower_num > 6)
				lower_num = 6;

			double green = inferno_g_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num+1] - inferno_values[lower_num])) * (inferno_g_range[lower_num+1] - inferno_g_range[lower_num]));
			double red = inferno_r_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_r_range[lower_num + 1] - inferno_r_range[lower_num]));
			double blue = inferno_b_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_b_range[lower_num + 1] - inferno_b_range[lower_num]));
			glColor3f(std::min(1.0,red), std::max(0.0,green), std::max(0.0, blue));
			glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
		}
	}

	_dws[window_index].max_mass = max_m;
	_dws[window_index].min_mass = min_m;

	glEnd();

	// Print real time and sim time
	if (SHOW_TEXT) {
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
		while (pos < nice_max_h) {
			if (std::abs(pos) < 0.0000000001)
				pos = 0.0;

			glColor3f(1.0, 1.0, 1.0);
			glRasterPos2f(-1.0, 2 * ((pos - (mesh_min_h + ((mesh_max_h - mesh_min_h) / 2.0))) / (mesh_max_h - mesh_min_h)));

			std::stringstream stream;
			stream << std::setprecision(3) << pos;
			t = stream.str();
			c_string = t.c_str();
			len = (int)strlen(c_string);
			for (i = 0; i < len; i++) {
				glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
			}
			pos += h_step;
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

bool gluInvertMatrix(const GLfloat m[16], GLfloat invOut[16])
{
	GLfloat inv[16], det;
	int i;

	inv[0] = m[5] * m[10] * m[15] -
		m[5] * m[11] * m[14] -
		m[9] * m[6] * m[15] +
		m[9] * m[7] * m[14] +
		m[13] * m[6] * m[11] -
		m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] +
		m[4] * m[11] * m[14] +
		m[8] * m[6] * m[15] -
		m[8] * m[7] * m[14] -
		m[12] * m[6] * m[11] +
		m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] -
		m[4] * m[11] * m[13] -
		m[8] * m[5] * m[15] +
		m[8] * m[7] * m[13] +
		m[12] * m[5] * m[11] -
		m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] +
		m[4] * m[10] * m[13] +
		m[8] * m[5] * m[14] -
		m[8] * m[6] * m[13] -
		m[12] * m[5] * m[10] +
		m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] +
		m[1] * m[11] * m[14] +
		m[9] * m[2] * m[15] -
		m[9] * m[3] * m[14] -
		m[13] * m[2] * m[11] +
		m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] -
		m[0] * m[11] * m[14] -
		m[8] * m[2] * m[15] +
		m[8] * m[3] * m[14] +
		m[12] * m[2] * m[11] -
		m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] +
		m[0] * m[11] * m[13] +
		m[8] * m[1] * m[15] -
		m[8] * m[3] * m[13] -
		m[12] * m[1] * m[11] +
		m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] -
		m[0] * m[10] * m[13] -
		m[8] * m[1] * m[14] +
		m[8] * m[2] * m[13] +
		m[12] * m[1] * m[10] -
		m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] -
		m[1] * m[7] * m[14] -
		m[5] * m[2] * m[15] +
		m[5] * m[3] * m[14] +
		m[13] * m[2] * m[7] -
		m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] +
		m[0] * m[7] * m[14] +
		m[4] * m[2] * m[15] -
		m[4] * m[3] * m[14] -
		m[12] * m[2] * m[7] +
		m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] -
		m[0] * m[7] * m[13] -
		m[4] * m[1] * m[15] +
		m[4] * m[3] * m[13] +
		m[12] * m[1] * m[7] -
		m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] +
		m[0] * m[6] * m[13] +
		m[4] * m[1] * m[14] -
		m[4] * m[2] * m[13] -
		m[12] * m[1] * m[6] +
		m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] +
		m[1] * m[7] * m[10] +
		m[5] * m[2] * m[11] -
		m[5] * m[3] * m[10] -
		m[9] * m[2] * m[7] +
		m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] -
		m[0] * m[7] * m[10] -
		m[4] * m[2] * m[11] +
		m[4] * m[3] * m[10] +
		m[8] * m[2] * m[7] -
		m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] +
		m[0] * m[7] * m[9] +
		m[4] * m[1] * m[11] -
		m[4] * m[3] * m[9] -
		m[8] * m[1] * m[7] +
		m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] -
		m[0] * m[6] * m[9] -
		m[4] * m[1] * m[10] +
		m[4] * m[2] * m[9] +
		m[8] * m[1] * m[6] -
		m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

	if (det == 0)
		return false;

	det = 1.0 / det;

	for (i = 0; i < 16; i++)
		invOut[i] = inv[i] * det;

	return true;
}

GLfloat modulof(GLfloat a, GLfloat b) {
	GLfloat r = std::fmod(a,b);
	return r < 0 ? r + b : r;
}

GLfloat camDist(std::vector<double>& point1, std::vector<double>& point2) {
	return std::sqrt(((point2[0]-point1[0])* (point2[0] - point1[0])) + ((point2[1] - point1[1]) * (point2[1] - point1[1])) + ((point2[2] - point1[2]) * (point2[2] - point1[2])));
}

template <typename T>
std::vector<unsigned int> sort_indexes(const std::vector<T>& v) {

	// initialize original index locations
	std::vector<unsigned int> idx(v.size());
	iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
	std::stable_sort(idx.begin(), idx.end(),
		[&v](unsigned int i1, unsigned int i2) {return v[i1] > v[i2]; });

	return idx;
}

double dot_prod(std::vector<double>& vec1, std::vector<double>& vec2) {
	double sum = 0.0;
	for (unsigned int i = 0; i < vec1.size(); i++) {
		sum += vec1[i] * vec2[i];
	}

	return sum;
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

	milliseconds real_time = duration_cast<milliseconds>(
		system_clock::now().time_since_epoch());
	milliseconds time_elapsed = real_time - start_time;

	int window_index = 0;
	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter) {
		if (iter->second._window_index == glutGetWindow())
			window_index = iter->first;
	}

	if (pgupPressed) {
		_dws[window_index].dim_select = modulo((_dws[window_index].dim_select + 1) , (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions()));
		pgupPressed = false;
	}

	if (pgdnPressed) {
		_dws[window_index].dim_select = modulo((_dws[window_index].dim_select - 1) , (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions()));
		pgdnPressed = false;
	}

	if (upPressed) {
		_dws[window_index].rot_x += 1.5f;
	}

	if (downPressed) {
		_dws[window_index].rot_x -= 1.5f;
	}

	if (leftPressed) {
		_dws[window_index].rot_y += 1.5f;
	}

	if (rightPressed) {
		_dws[window_index].rot_y -= 1.5f;
	}

	bool update_draw_order = leftPressed || upPressed || downPressed || rightPressed;

	glViewport(0, 0, _dws[window_index].width, _dws[window_index].height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(45.0f, (GLfloat)_dws[window_index].width / (GLfloat)_dws[window_index].height, 0.1f, 50.0f);

	gluLookAt(  0, -0.75,-1.5, // distance from centre of the cube
				0, 0, 0,
				0, 1, 0); //Orient the camera

	std::vector<double> camPos = { 0,0,-1.5,1.0 };

	//glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//if (time_elapsed.count() % 10 != 0)
	// 	return;

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	//glClear(GL_COLOR_BUFFER_BIT);

	// **** used for 3D ****
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat x_rot = modulof(25 + _dws[window_index].rot_x, 360) - 45;
	glRotatef(x_rot, 1.0f, 0.0f, 0.0f);
	GLfloat y_rot = modulof(-155 + _dws[window_index].rot_y, 360) - 45;
	glRotatef(y_rot, 0.0f, 1.0f, 0.0f);

	GLfloat inv[16];
	GLfloat viewmat[16];

	glGetFloatv(GL_MODELVIEW_MATRIX, viewmat);

	gluInvertMatrix(viewmat, inv);

	//std::vector<double> a = { inv[0],inv[1],inv[2],inv[3] };
	//std::vector<double> b = { inv[4],inv[5],inv[6],inv[7] };
	//std::vector<double> c = { inv[8],inv[9],inv[10],inv[11] };
	//std::vector<double> d = { inv[12],inv[13],inv[14],inv[15] };
	std::vector<double> a = { inv[0],inv[4],inv[8],inv[12] };
	std::vector<double> b = { inv[1],inv[5],inv[9],inv[13] };
	std::vector<double> c = { inv[2],inv[5],inv[10],inv[14] };
	std::vector<double> d = { inv[3],inv[7],inv[11],inv[15] };
	std::vector<std::vector<double>> dinv = { a,b,c,d };

	std::vector<double> inverted_cam_pos = mat_mult(dinv, camPos);

	std::vector<std::string> labels4(4);
	labels4[0] = std::string("h");
	labels4[1] = std::string("n");
	labels4[2] = std::string("m");
	labels4[3] = std::string("v");

	glColor3f(1.0, 1.0, 1.0);
	glRasterPos3f(-0.5f, 0.55f, -0.5f);
	int len, i;

	if (SHOW_TEXT) {

		std::string t = std::string("u");
		if (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions() == 4)
			t = labels4[modulo(1 + _dws[window_index].dim_select, _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions())];
		const char* c_string = t.c_str();
		len = (int)strlen(c_string);
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c_string[i]);
		}

		glColor3f(1.0, 1.0, 1.0);
		glRasterPos3f(-0.5f, -0.5f, 0.6f);
		t = std::string("w");
		if (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions() == 4)
			t = labels4[modulo(2 + _dws[window_index].dim_select, _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions())];
		c_string = t.c_str();
		len = (int)strlen(c_string);
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c_string[i]);
		}

		glColor3f(1.0, 1.0, 1.0);
		glRasterPos3f(0.55f, -0.5f, -0.5f);
		t = std::string("v");
		if (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions() == 4)
			t = labels4[modulo(3 + _dws[window_index].dim_select, _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions())];
		c_string = t.c_str();
		len = (int)strlen(c_string);
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c_string[i]);
		}

		// Draw lines

		glLineWidth(2.0f);
		glBegin(GL_LINES);

		// y axis
		glVertex3f(-0.5f, -0.5f, -0.5f);
		glVertex3f(-0.5f, 0.5f, -0.5f);

		glVertex3f(-0.5f, 0.5f, -0.5f);
		glVertex3f(-0.5f, 0.48f, -0.48f);

		glVertex3f(-0.5f, 0.5f, -0.5f);
		glVertex3f(-0.48f, 0.48f, -0.5f);

		glVertex3f(-0.5f, 0.5f, -0.5f);
		glVertex3f(-0.5f, 0.48f, -0.52f);

		glVertex3f(-0.5f, 0.5f, -0.5f);
		glVertex3f(-0.52f, 0.48f, -0.5f);

		// x axis
		glVertex3f(-0.5f, -0.5f, -0.5f);
		glVertex3f(0.5f, -0.5f, -0.5f);

		glVertex3f(0.5f, -0.5f, -0.5f);
		glVertex3f(0.48f, -0.48f, -0.5f);

		glVertex3f(0.5f, -0.5f, -0.5f);
		glVertex3f(0.48f, -0.5f, -0.48f);

		glVertex3f(0.5f, -0.5f, -0.5f);
		glVertex3f(0.48f, -0.5f, -0.52f);

		glVertex3f(0.5f, -0.5f, -0.5f);
		glVertex3f(0.48f, -0.52f, -0.5f);

		// z axis
		glVertex3f(-0.5f, -0.5f, -0.5f);
		glVertex3f(-0.5f, -0.5f, 0.5f);

		glVertex3f(-0.5f, -0.5f, 0.5f);
		glVertex3f(-0.48f, -0.5f, 0.48f);

		glVertex3f(-0.5f, -0.5f, 0.5f);
		glVertex3f(-0.5f, -0.48f, 0.48f);

		glVertex3f(-0.5f, -0.5f, 0.5f);
		glVertex3f(-0.5f, -0.52f, 0.48f);

		glVertex3f(-0.5f, -0.5f, 0.5f);
		glVertex3f(-0.52f, -0.5f, 0.48f);

		glEnd();
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);

	glBegin(GL_QUADS);

	Mesh m = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index];

	double max = 0.5;
	double min = 0.0;

	double mesh_min_v = _dws[window_index].mesh_min_v;
	double mesh_max_v = _dws[window_index].mesh_max_v;
	double mesh_min_h = _dws[window_index].mesh_min_h;
	double mesh_max_h = _dws[window_index].mesh_max_h;

	// Display 3D mass

	unsigned int size_x = m.getGridResolutionByDimension(modulo(-1 + _dws[window_index].dim_select, _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions()));
	unsigned int size_y = m.getGridResolutionByDimension(modulo(-2 + _dws[window_index].dim_select, _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions()));
	unsigned int size_z = m.getGridResolutionByDimension(modulo(-3 + _dws[window_index].dim_select, _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions()));

	// Draw a Cube
	if (SHOW_3D_SPACE) {

		double cell_x = 0.0;
		double cell_y = 0.0;
		double cell_z = 0.0;

		double half_cell_x_width = 0.5;
		double half_cell_y_width = 0.5;
		double half_cell_z_width = 0.5;

		std::vector<double> p1 = { cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p2 = { cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p3 = { cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p4 = { cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p5 = { cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		std::vector<double> p6 = { cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		std::vector<double> p7 = { cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		std::vector<double> p8 = { cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		//glColor4f(0.0, 0.0, 0.0, 1);
		glColor4f(0.3, 0.3, 0.3, 0.1);
		// front face
		glVertex3f(p1[0], p1[1], p1[2]);
		glVertex3f(p2[0], p2[1], p2[2]);
		glVertex3f(p3[0], p3[1], p3[2]);
		glVertex3f(p4[0], p4[1], p4[2]);

		// back face
		glVertex3f(p6[0], p6[1], p6[2]);
		glVertex3f(p5[0], p5[1], p5[2]);
		glVertex3f(p8[0], p8[1], p8[2]);
		glVertex3f(p7[0], p7[1], p7[2]);

		glColor4f(0.35, 0.35, 0.35, 1);
		// right face
		glVertex3f(p4[0], p4[1], p4[2]);
		glVertex3f(p3[0], p3[1], p3[2]);
		glVertex3f(p7[0], p7[1], p7[2]);
		glVertex3f(p8[0], p8[1], p8[2]);

		// left face
		glVertex3f(p6[0], p6[1], p6[2]);
		glVertex3f(p2[0], p2[1], p2[2]);
		glVertex3f(p1[0], p1[1], p1[2]);
		glVertex3f(p5[0], p5[1], p5[2]);

		glColor4f(0.4, 0.4, 0.4, 1);
		//// top face
		glVertex3f(p2[0], p2[1], p2[2]);
		glVertex3f(p6[0], p6[1], p6[2]);
		glVertex3f(p7[0], p7[1], p7[2]);
		glVertex3f(p3[0], p3[1], p3[2]);

		//// bottom face
		glVertex3f(p5[0], p5[1], p5[2]);
		glVertex3f(p1[0], p1[1], p1[2]);
		glVertex3f(p4[0], p4[1], p4[2]);
		glVertex3f(p8[0], p8[1], p8[2]);
	}

	// Do some crude ordering so that the alpha blend looks ok from all angles
	std::vector<unsigned int> res_order(3);
	res_order[0] = 0; //w
	res_order[1] = 1; //u
	res_order[2] = 2; //v

	std::vector<unsigned int> vals_start(3);
	vals_start[0] = 0;
	vals_start[1] = 0;
	vals_start[2] = 0;

	std::vector<unsigned int> vals_end(3);
	vals_end[0] = size_z;
	vals_end[1] = size_y;
	vals_end[2] = size_x;

	std::vector<unsigned int> vals_diff(3);
	vals_diff[0] = 1;
	vals_diff[1] = 1;
	vals_diff[2] = 1;

	std::vector<bool> reverse(3);
	reverse[0] = false;
	reverse[1] = false;
	reverse[2] = false;

	std::vector<unsigned int> vals(3);
	vals[res_order[0]] = vals_start[res_order[0]];
	vals[res_order[1]] = vals_start[res_order[1]];
	vals[res_order[2]] = vals_start[res_order[2]];

	// Perfect but slow depth ordering
	int num_cells = size_x * size_y * size_z;
	std::vector<unsigned int> cell_inds(num_cells);
	std::vector<GLfloat> depths(num_cells);
	int current_cell_ind = 0;

	// Crude but fast ordering
	/*for (vals[res_order[0]] = vals_start[res_order[0]]; vals[res_order[0]] < vals_end[res_order[0]]; vals[res_order[0]] += vals_diff[res_order[0]]) {
		for (vals[res_order[1]] = vals_start[res_order[1]]; vals[res_order[1]] < vals_end[res_order[1]]; vals[res_order[1]] += vals_diff[res_order[1]]) {
			for (vals[res_order[2]] = vals_start[res_order[2]]; vals[res_order[2]] < vals_end[res_order[2]]; vals[res_order[2]] += vals_diff[res_order[2]]) {
				unsigned int i_r = vals[0];
				if (reverse[0])
					i_r = vals_end[0] - 1 - vals[0];
				unsigned int k_r = vals[1];
				if (reverse[1])
					k_r = vals_end[1] - 1 - vals[1];
				unsigned int j_r = vals[2];
				if (reverse[2])
					j_r = vals_end[2] - 1 - vals[2];

				std::vector<unsigned int> coords3(3);
				coords3[0] = i_r;
				coords3[1] = k_r;
				coords3[2] = j_r;

				unsigned int idx = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getIndexOfCoords(coords3) + _dws[window_index]._system->Offsets()[_dws[window_index]._mesh_index];
				cell_inds[current_cell_ind] = idx;
				current_cell_ind++;
			}
		}
	}*/

	for (vals[res_order[0]] = vals_start[res_order[0]]; vals[res_order[0]] < vals_end[res_order[0]]; vals[res_order[0]] += vals_diff[res_order[0]]) {
		for (vals[res_order[1]] = vals_start[res_order[1]]; vals[res_order[1]] < vals_end[res_order[1]]; vals[res_order[1]] += vals_diff[res_order[1]]) {
			for (vals[res_order[2]] = vals_start[res_order[2]]; vals[res_order[2]] < vals_end[res_order[2]]; vals[res_order[2]] += vals_diff[res_order[2]]) {
				unsigned int i_r = vals[0];
				if (reverse[0])
					i_r = vals_end[0] - 1 - vals[0];
				unsigned int k_r = vals[1];
				if (reverse[1])
					k_r = vals_end[1] - 1 - vals[1];
				unsigned int j_r = vals[2];
				if (reverse[2])
					j_r = vals_end[2] - 1 - vals[2];

				double cell_x = -0.5 + j_r * (1.0 / vals_end[2]);
				double cell_y = -0.5 + i_r * (1.0 / vals_end[0]);
				double cell_z = -0.5 + k_r * (1.0 / vals_end[1]);
				std::vector<double> cell_point = { cell_x, cell_y, cell_z };

				depths[current_cell_ind] = camDist(cell_point, inverted_cam_pos);
				current_cell_ind++;
			}
		}
	}
	
	cell_inds = sort_indexes(depths);

	std::vector<double> inferno_r_range(8);
	std::vector<double> inferno_g_range(8);
	std::vector<double> inferno_b_range(8);
	std::vector<double> inferno_values(8);

	for (unsigned int i = 0; i < 8; i++) {
		inferno_values[i] = (1.0 / (double)8) * (double)i;
	}

	inferno_r_range[0] = (double)0 / (double)255;
	inferno_g_range[0] = (double)0 / (double)255;
	inferno_b_range[0] = (double)4 / (double)255;

	inferno_r_range[1] = (double)40 / (double)255;
	inferno_g_range[1] = (double)11 / (double)255;
	inferno_b_range[1] = (double)84 / (double)255;

	inferno_r_range[2] = (double)101 / (double)255;
	inferno_g_range[2] = (double)21 / (double)255;
	inferno_b_range[2] = (double)110 / (double)255;

	inferno_r_range[3] = (double)159 / (double)255;
	inferno_g_range[3] = (double)42 / (double)255;
	inferno_b_range[3] = (double)99 / (double)255;

	inferno_r_range[4] = (double)212 / (double)255;
	inferno_g_range[4] = (double)72 / (double)255;
	inferno_b_range[4] = (double)66 / (double)255;

	inferno_r_range[5] = (double)245 / (double)255;
	inferno_g_range[5] = (double)125 / (double)255;
	inferno_b_range[5] = (double)21 / (double)255;

	inferno_r_range[6] = (double)250 / (double)255;
	inferno_g_range[6] = (double)193 / (double)255;
	inferno_b_range[6] = (double)39 / (double)255;

	inferno_r_range[7] = (double)252 / (double)255;
	inferno_g_range[7] = (double)255 / (double)255;
	inferno_b_range[7] = (double)164 / (double)255;
	
	double max_m = -9999999;
	double min_m = 9999999;
	for (unsigned int idx : cell_inds) {

		unsigned int i_r = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getCoordsOfIndex(idx)[0];
		unsigned int k_r = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getCoordsOfIndex(idx)[1];
		unsigned int j_r = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getCoordsOfIndex(idx)[2];

		double cell_mass = 0.0;
		double mass = 0.0;

		if (_dws[window_index]._system->FiniteSizeNumObjects()[_dws[window_index]._mesh_index] == 0) {
			// Sum up all mass above the third dimension as we can't display it (this is effectively a marginal)
			// for the sake of speed here, just assume that this is 4D - if you're in the future and working on 5D or greater
			// then congratulations on solving the dimensionality problem or possibly quantum computing.
			if (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions() == 4) {

				i_r = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getCoordsOfIndex(idx)[3];
				k_r = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getCoordsOfIndex(idx)[2];
				j_r = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getCoordsOfIndex(idx)[1];

				std::vector<unsigned int> coords4(4);
				coords4[0 + _dws[window_index].dim_select] = 0;
				coords4[modulo(1 + _dws[window_index].dim_select, 4)] = i_r;
				coords4[modulo(2 + _dws[window_index].dim_select, 4)] = k_r;
				coords4[modulo(3 + _dws[window_index].dim_select, 4)] = j_r;

				unsigned int doo = 1;
				for (int d = 3; d >= _dws[window_index].dim_select+1; d--)
					doo *= _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridResolutionByDimension(d);

				unsigned int idx = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getIndexOfCoords(coords4) + _dws[window_index]._system->Offsets()[_dws[window_index]._mesh_index];
				for (unsigned int c = 0; c < _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridResolutionByDimension(_dws[window_index].dim_select); c++) {
					unsigned int index = idx + (c * doo);
					cell_mass += _dws[window_index]._system->Mass()[index];
				}
			}
			else {
				cell_mass = _dws[window_index]._system->Mass()[idx];
			}

			if (cell_mass < 0.000000001) continue; // skip if mass is basically nothing

			double cell_area = std::abs(m.Quad(0, 0).SignedArea());

			if (!log_scale) {
				if (max_m < 1e-6 + cell_mass / cell_area)
					max_m = 1e-6 + cell_mass / cell_area;
				if (min_m >= 1e-6 + cell_mass / cell_area)
					min_m = 1e-6 + cell_mass / cell_area;
			}
			else {
				if (max_m < log10(1e-6 + cell_mass / cell_area))
					max_m = log10(1e-6 + cell_mass / cell_area);
				if (min_m >= log10(1e-6 + cell_mass / cell_area))
					min_m = log10(1e-6 + cell_mass / cell_area);
			}

			if (!log_scale) {
				mass = (cell_mass / cell_area - _dws[window_index].min_mass) / (_dws[window_index].max_mass - _dws[window_index].min_mass);
			}
			else {
				mass = (log10(cell_mass / cell_area) - _dws[window_index].min_mass) / (_dws[window_index].max_mass - _dws[window_index].min_mass);
			}
		}
		else {
			unsigned int count = 0;
			if (_dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridNumDimensions() == 4) {
				std::vector<unsigned int> coords4(4);
				coords4[0 + _dws[window_index].dim_select] = 0;
				coords4[modulo(1 + _dws[window_index].dim_select, 4)] = i_r;
				coords4[modulo(2 + _dws[window_index].dim_select, 4)] = k_r;
				coords4[modulo(3 + _dws[window_index].dim_select, 4)] = j_r;

				unsigned int doo = 1;
				for (int d = 3; d >= _dws[window_index].dim_select + 1; d--)
					doo *= _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridResolutionByDimension(d);

				unsigned int idx = _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getIndexOfCoords(coords4) + _dws[window_index]._system->Offsets()[_dws[window_index]._mesh_index];
				for (unsigned int c = 0; c < _dws[window_index]._system->MeshObjects()[_dws[window_index]._mesh_index].getGridResolutionByDimension(_dws[window_index].dim_select); c++) {
					unsigned int index = idx + (c * doo);
					count += _dws[window_index]._system->_vec_cells_to_objects[index].size();
				}
			}
			else {
				count = _dws[window_index]._system->_vec_cells_to_objects[idx].size();
			}

			if (count > 0)
				mass = 0.5 + 1000.0 * ((double)count / (double)_dws[window_index]._system->FiniteSizeNumObjects()[_dws[window_index]._mesh_index]);
			else
				continue;
		}

		unsigned int lower_num = int(mass * 8.0);
		if (lower_num > 6)
			lower_num = 6;

		double green = inferno_g_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_g_range[lower_num + 1] - inferno_g_range[lower_num]));
		double red = inferno_r_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_r_range[lower_num + 1] - inferno_r_range[lower_num]));
		double blue = inferno_b_range[lower_num] + (((mass - inferno_values[lower_num]) / (inferno_values[lower_num + 1] - inferno_values[lower_num])) * (inferno_b_range[lower_num + 1] - inferno_b_range[lower_num]));
		//glColor3f(std::min(1.0, red), std::max(0.0, green), std::max(0.0, blue));
					
		if (mass < 0.00000001) {
			continue; // glColor4f(1.0, 1.0, 1.0, 0.02);
		}
		else if (mass > 0.00000001) {
			glColor4f(std::min(1.0, red), std::max(0.0, green), std::max(0.0, blue), mass);
		}
		else {
			continue;
		}

		double cell_x = -0.5 + j_r * (1.0 / vals_end[2]);
		double cell_y = -0.5 + i_r * (1.0 / vals_end[0]);
		double cell_z = -0.5 + k_r * (1.0 / vals_end[1]);

		double half_cell_x_width = (0.5 / vals_end[2]);
		double half_cell_y_width = (0.5 / vals_end[0]);
		double half_cell_z_width = (0.5 / vals_end[1]);

		std::vector<double> p1 = { cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p2 = { cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p3 = { cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p4 = { cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z + half_cell_z_width, 1.0 };
		std::vector<double> p5 = { cell_x - half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		std::vector<double> p6 = { cell_x - half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		std::vector<double> p7 = { cell_x + half_cell_x_width, cell_y + half_cell_y_width, cell_z - half_cell_z_width, 1.0 };
		std::vector<double> p8 = { cell_x + half_cell_x_width, cell_y - half_cell_y_width, cell_z - half_cell_z_width, 1.0 };

		std::vector<double> cam_dir = {(cell_x - inverted_cam_pos[0]),(cell_y - inverted_cam_pos[1]),(cell_z - inverted_cam_pos[2])};
		// front face
		std::vector<double> dir = {0,0,-1};
		if (dot_prod(dir, cam_dir) > 0) {
			glVertex3f(p2[0], p2[1], p2[2]);
			glVertex3f(p1[0], p1[1], p1[2]);
			glVertex3f(p4[0], p4[1], p4[2]);
			glVertex3f(p3[0], p3[1], p3[2]);
		}

		// back face
		dir = { 0,0,1 };
		if (dot_prod(dir, cam_dir) > 0) {
			glVertex3f(p5[0], p5[1], p5[2]);
			glVertex3f(p6[0], p6[1], p6[2]);
			glVertex3f(p7[0], p7[1], p7[2]);
			glVertex3f(p8[0], p8[1], p8[2]);
		}

		// right face
		dir = { -1,0,0 };
		if (dot_prod(dir, cam_dir) > 0) {
			glVertex3f(p3[0], p3[1], p3[2]);
			glVertex3f(p4[0], p4[1], p4[2]);
			glVertex3f(p8[0], p8[1], p8[2]);
			glVertex3f(p7[0], p7[1], p7[2]);
		}

		// left face
		dir = { 1,0,0 };
		if (dot_prod(dir, cam_dir) > 0) {
			glVertex3f(p2[0], p2[1], p2[2]);
			glVertex3f(p6[0], p6[1], p6[2]);
			glVertex3f(p5[0], p5[1], p5[2]);
			glVertex3f(p1[0], p1[1], p1[2]);
		}

		//// top face
		dir = { 0,-1,0 };
		if (dot_prod(dir, cam_dir) > 0) {
			glVertex3f(p6[0], p6[1], p6[2]);
			glVertex3f(p2[0], p2[1], p2[2]);
			glVertex3f(p3[0], p3[1], p3[2]);
			glVertex3f(p7[0], p7[1], p7[2]);
		}

		//// bottom face
		dir = { 0,1,0 };
		if (dot_prod(dir, cam_dir) > 0) {
			glVertex3f(p1[0], p1[1], p1[2]);
			glVertex3f(p5[0], p5[1], p5[2]);
			glVertex3f(p8[0], p8[1], p8[2]);
			glVertex3f(p4[0], p4[1], p4[2]);
		}
	}

	_dws[window_index].max_mass = max_m;
	_dws[window_index].min_mass = min_m;

	glEnd();

	// Print real time and sim time

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, 1.0, 1.0, 0);
	glViewport(0, 0, 1.0, 1.0);
	glLoadIdentity();

	double sim_time = 0.0;

	sim_time = _current_sim_it * _time_step;

	if (SHOW_TEXT) {


		glColor3f(1.0, 1.0, 1.0);
		glRasterPos2f(0.3, 0.9);
		int text_len, text_i;
		std::string text_v = std::string("Sim Time (s) : ") + std::to_string(sim_time);
		const char* c_string_v = text_v.c_str();
		text_len = (int)strlen(c_string_v);
		for (text_i = 0; text_i < text_len; text_i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string_v[text_i]);
		}

	}

	glPopMatrix();

	glutSwapBuffers();

	if (write_frames)
		writeFrame(window_index, _current_sim_it);
}

void Display::writeFrame(unsigned int system, long frame_num){
	//This prevents the images getting padded
 	// when the width multiplied by 3 is not a multiple of 4
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  // width * height * 3
  int nSize = WINDOW_WIDTH * WINDOW_HEIGHT *3;
  // First let's create our buffer, 3 channels per Pixel
  char* dataBuffer = (char*)malloc(nSize*sizeof(char));

  if (!dataBuffer) return;

   // Let's fetch them from the backbuffer
   // We request the pixels in GL_BGR format, thanks to Berzeger for the tip
  //  glReadPixels((GLint)0, (GLint)0,
	// 	(GLint)w, (GLint)h,
	// 	 GL_BGR, GL_UNSIGNED_BYTE, dataBuffer);
	 glReadPixels((GLint)0, (GLint)0,
	 (GLint)WINDOW_WIDTH, (GLint)WINDOW_HEIGHT,
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
	 unsigned char header[6] = { WINDOW_WIDTH %256,WINDOW_WIDTH /256,
						WINDOW_HEIGHT %256,WINDOW_HEIGHT /256,
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

	int window_index = 0;
	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter) {
		if (iter->second._window_index == glutGetWindow())
			window_index = iter->first;
	}

	_dws[window_index].width = width;
	_dws[window_index].height = height;
}

void Display::init() const {
	
}

void Display::init_3d() const {
}

void Display::update() {
}

void Display::update_3d() {
}

void Display::keyboard_3d_down(int key, int _x, int _y) {

	if (key == GLUT_KEY_UP)
		upPressed = true;

	if (key == GLUT_KEY_DOWN)
		downPressed = true;

	if (key == GLUT_KEY_LEFT)
		leftPressed = true;

	if (key == GLUT_KEY_RIGHT)
		rightPressed = true;

	if (key == GLUT_KEY_PAGE_UP)
		pgupPressed = true;

	if (key == GLUT_KEY_PAGE_DOWN)
		pgdnPressed = true;
}

void Display::keyboard_3d_up(int key, int _x, int _y) {

	if (key == GLUT_KEY_UP)
		upPressed = false;

	if (key == GLUT_KEY_DOWN)
		downPressed = false;

	if (key == GLUT_KEY_LEFT)
		leftPressed = false;

	if (key == GLUT_KEY_RIGHT)
		rightPressed = false;

	if (key == GLUT_KEY_PAGE_UP)
		pgupPressed = false;

	if (key == GLUT_KEY_PAGE_DOWN)
		pgdnPressed = false;
}


void Display::updateDisplay(long current_sim_it) {
	if (Display::getInstance()->_nodes_to_display.size() == 0)
		return;

	int time;
	time = glutGet(GLUT_ELAPSED_TIME);
	Display::getInstance()->_current_sim_it = current_sim_it;
	lastTime = time;
	std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
void Display::setDisplayNodes(std::vector<MPILib::NodeId> nodes_to_display) const {
	Display::getInstance()->_nodes_to_display = nodes_to_display;
}

void Display::animate(bool _write_frames, std::vector<MPILib::NodeId> display_nodes, double time_step) const {
	Display::getInstance()->_nodes_to_display = display_nodes;
	animate(_write_frames, time_step);
}

void Display::animate(bool _write_frames,  double time_step) const{

	if (Display::getInstance()->_nodes_to_display.size() == 0)
		return;

	Display::getInstance()->write_frames = _write_frames;
	Display::getInstance()->_time_step = time_step;

	char* arv[] = {"Miind"};
	int count = 1;
	glutInit(&count, arv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(0, 0);
	glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);

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
			glutSpecialFunc(Display::stat_keyboard_3d_down);
			glutSpecialUpFunc(Display::stat_keyboard_3d_up);
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
