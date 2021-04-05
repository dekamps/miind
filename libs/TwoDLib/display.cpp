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

unsigned int Display::addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys) {
	return addOdeSystem(nid, sys, 0);
}

unsigned int Display::addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys, unsigned int mesh_index) {
	unsigned int index = _dws.size();

	DisplayWindow window;
	window._system = sys;
	window._mesh_index = mesh_index;

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
			if (_dws[window_index]._system->Mass()[idx] / cell_area != 0 && _dws[window_index]._system->_num_objects == 0)
				mass = std::min(1.0, std::max(0.0, (log10(_dws[window_index]._system->Mass()[idx] / cell_area) - min) / (max - min)));
			else
				if (_dws[window_index]._system->_vec_cells_to_objects[idx].size() > 0)
					mass = 1.0;

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
	// **** used for 3D ****
	// if (height == 0)
	// {
	// 	height = 1;
	// }

	// glViewport(0, 0, width, height);
	// glMatrixMode(GL_PROJECTION);
	// glLoadIdentity();
  //
	// gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
  //
	// glMatrixMode(GL_MODELVIEW);
	// glLoadIdentity();

	glViewport(0, 0, width, height);
	glLoadIdentity();
}

void Display::init() const {
	// **** used for 3D ****
	//gluPerspective(45, 1, 2, 10);
	//glEnable(GL_DEPTH_TEST);
	glClearColor(0.0f, 0.0f, 0.2f, 0.0f);
}

void Display::update() {
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
		Display::getInstance()->_dws[Display::getInstance()->_nodes_to_display[id]]._window_index = glutCreateWindow("Miind2D");
		glutDisplayFunc(Display::stat_display);
		glutReshapeFunc(Display::stat_scene);
		glutIdleFunc(Display::stat_update);
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
