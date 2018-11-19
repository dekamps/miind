#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iomanip>
#include <sstream>
#include <omp.h>
#include <math.h>
#include <boost/filesystem.hpp>

#include "display.hpp"
#include "include/glm/glm.hpp"
#include <MPILib/include/MPINetworkCode.hpp>

using namespace TwoDLib;
using namespace std::chrono;

Display* Display::disp = 0;

Display::Display(){
	lastTime = 0;
	write_frames = false;
	start_time = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
	_dws = std::map<MPILib::NodeId, DisplayWindow>();
	// When using OpenMP, all cores are reserved for simulation. For displaying,
	// we need to set aside a thread/core otherwise display gets interleaved and
	// is super slow.
	int num_threads = omp_get_max_threads();
	if (num_threads > 1)
		omp_set_num_threads(num_threads-1);
}

Display::~Display(){
	if (glutGetWindow()){
		Display::getInstance()->updateDisplay();
		glutDestroyWindow(glutGetWindow());
	}
	glutExit();
}

unsigned int Display::addOdeSystem(MPILib::NodeId nid, Ode2DSystem* sys, std::mutex *mu) {
	unsigned int index = _dws.size();

	DisplayWindow window;
	window._system = sys;
	window._read_mutex = mu;


	// Find extent of mesh to normalise to screen size

	Mesh m = sys->MeshObject();

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

	glClear(GL_COLOR_BUFFER_BIT);

	// **** used for 3D ****
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glPushMatrix();
	// glTranslatef(0, 0, -1);

	glBegin(GL_QUADS);

	Mesh m = _dws[window_index]._system->MeshObject();

	LockMutex(window_index);

	double max = 0.0;
	for (int i=0; i<_dws[window_index]._system->Mass().size(); i++)
		if (max < log10(_dws[window_index]._system->Mass()[i] + 0.000006))
			max = log10(_dws[window_index]._system->Mass()[i] + 0.000006);

	double min = 1.0;
	for (int i=0; i<_dws[window_index]._system->Mass().size(); i++)
		if (log10(_dws[window_index]._system->Mass()[i] + 0.000006) < min)
			min = log10(_dws[window_index]._system->Mass()[i] + 0.000006);

	double mesh_min_v = _dws[window_index].mesh_min_v;
	double mesh_max_v = _dws[window_index].mesh_max_v;
	double mesh_min_h = _dws[window_index].mesh_min_h;
	double mesh_max_h = _dws[window_index].mesh_max_h;

	for(unsigned int i = 0; i<m.NrStrips(); i++){
		for(unsigned int j = 0; j<m.NrCellsInStrip(i); j++) {
			Quadrilateral q = m.Quad(i,j);
			unsigned int idx = _dws[window_index]._system->Map(i,j);
			double mass = (log10(_dws[window_index]._system->Mass()[idx] + 0.000006) - min) / (max-min);
			vector<Point> ps = q.Points();

			glColor3f(std::min(1.0,mass*2.0), std::max(0.0,((mass*2.0) - 1.0)), 0);
			glVertex2f(2*(ps[0][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[0][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[1][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[1][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[2][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[2][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
			glVertex2f(2*(ps[3][0]-(mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v), 2*(ps[3][1]-(mesh_min_h + ((mesh_max_h - mesh_min_h)/2)))/(mesh_max_h - mesh_min_h));
		}
	}


	UnlockMutex(window_index);

	glEnd();

	// Print real time and sim time

	double sim_time = 0.0;

	if(net)
		sim_time = net->getCurrentSimulationTime() * m.TimeStep();

	glColor3f( 1.0, 1.0, 1.0 );
  glRasterPos2f(0.3, 0.9);
  int len, i;
	std::string t = std::string("Sim Time (s) : ") + std::to_string( sim_time );
	const char* c_string = t.c_str();
  len = (int)strlen( c_string );
  for (i = 0; i < len; i++) {
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
  }

	double nice_min_h = (double)floor(mesh_min_h * 1) / 1.0;
	double nice_max_h = (double)ceil(mesh_max_h * 1) / 1.0;

	double nice_min_v = (double)floor(mesh_min_v * 0.01) / 0.01;
	double nice_max_v = (double)ceil(mesh_max_v * 0.01) / 0.01;

	double pos = 0;
	while(pos < nice_max_h){
		glColor3f( 1.0, 1.0, 1.0 );
	  glRasterPos2f(-1.0, 2*((pos - (mesh_min_h + ((mesh_max_h - mesh_min_h)/2.0)))/(mesh_max_h - mesh_min_h)));
		std::stringstream stream;
		stream << std::fixed << std:: setprecision(1) << pos;
		t = stream.str();
		c_string = t.c_str();
	  len = (int)strlen( c_string );
	  for (i = 0; i < len; i++) {
	    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
	  }
		pos +=  (nice_max_h - nice_min_h) / 10;
	}
	pos = 0;
	while(pos > nice_min_h){
		glColor3f( 1.0, 1.0, 1.0 );
	  glRasterPos2f(-1.0, 2*((pos - (mesh_min_h + ((mesh_max_h - mesh_min_h)/2.0)))/(mesh_max_h - mesh_min_h)));
		std::stringstream stream;
		stream << std::fixed << std:: setprecision(1) << pos;
		t = stream.str();
		c_string = t.c_str();
	  len = (int)strlen( c_string );
	  for (i = 0; i < len; i++) {
	    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
	  }
		pos -= (nice_max_h - nice_min_h) / 10;
	}

	pos = 0;
	while(pos < nice_max_v){
		glRasterPos2f(2*((pos - (mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v)), -1.0);
		std::stringstream stream2;
		stream2 << std::fixed << std:: setprecision(1) << pos;
		t = stream2.str();
		c_string = t.c_str();
		len = (int)strlen( c_string );
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
		}
		pos += (nice_max_v - nice_min_v) / 10;
	}
	pos = 0;
	while(pos > nice_min_v){
		glRasterPos2f(2*((pos - (mesh_min_v + ((mesh_max_v - mesh_min_v)/2.0)))/(mesh_max_v - mesh_min_v)), -1.0);
		std::stringstream stream2;
		stream2 << std::fixed << std:: setprecision(1) << pos;
		t = stream2.str();
		c_string = t.c_str();
		len = (int)strlen( c_string );
		for (i = 0; i < len; i++) {
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c_string[i]);
		}
		pos -= (nice_max_v - nice_min_v) / 10;
	}

	// **** used for 3D ****
	// glPopMatrix();
	glutSwapBuffers();
	glFlush();

	if(net && write_frames)
		writeFrame(window_index,net->getCurrentSimulationTime());
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
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
}

void Display::update() {
}

void Display::updateDisplay() {
	int time;
	time = glutGet(GLUT_ELAPSED_TIME);
	lastTime = time;
	//Sleep(50);
	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter){
		glutSetWindow(iter->second._window_index);
		glutPostRedisplay();
	}

	glutMainLoopEvent();

}

void Display::shutdown() const {
	glutExit();

	// just in case - unlock all mutexes.
	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter){
		disp->UnlockMutex(iter->first);
	}


	// Nice new line if we quit early.
	std::cout << "\n";
}

void Display::animate(bool _write_frames) const{

	Display::getInstance()->write_frames = _write_frames;

	char* arv[] = {"Miind"};
	int count = 1;
	glutInit(&count, arv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);

	for (std::map<MPILib::NodeId, DisplayWindow>::iterator iter = Display::getInstance()->_dws.begin(); iter != Display::getInstance()->_dws.end(); ++iter){
		iter->second._window_index = glutCreateWindow("Miind2D");
		glutDisplayFunc(Display::stat_display);
		glutReshapeFunc(Display::stat_scene);
		glutIdleFunc(Display::stat_update);
	}

	atexit(Display::stat_shutdown);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	init();
}

void Display::stat_runthreaded() {
	Display::getInstance()->animate(false);
	while((Display::getInstance()->close_display) && !(*Display::getInstance()->close_display)){

		if(!glutGetWindow()){
			glutExit();
			break;
		}

		Display::getInstance()->updateDisplay();
	}
}

void Display::processDraw(void) {
}
