#ifndef _CODE_LIBS_TWODLIB_DISPLAY_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_DISPLAY_INCLUDE_GUARD

#define LINUX // This #define is used in glew.c so it does not require installation
#include "include/GL/glew.h"
// The application does require glut to be installed
#include <GL/freeglut.h>

#include <string>
#include <chrono>
#include "Ode2DSystem.hpp"
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/DelayedConnection.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <mutex>

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;

namespace TwoDLib {

class Display{
public:

  static Display* getInstance() {
    if (!disp) {
      disp = new Display();
    }

    return disp;
  }

  void display(void);
  void scene(int width, int height);
  void init() const;
  void update();
  void shutdown() const;
  void animate() const;
  void processDraw(void);

  static void stat_display(void) {
    disp->display();
  }
  static void stat_scene(int width, int height) {
    disp->scene(width,height);
  }
  static void stat_update(void){
    disp->update();
  }
  static void stat_shutdown(void){
    disp->shutdown();
  }
  static void stat_runthreaded(void);

  void addOdeSystem(Ode2DSystem* sys) {
    _systems.push_back(sys);

    // Find extent of mesh to normalise to screen size

  	Mesh m = _systems[0]->MeshObject();

  	mesh_min_v = 0.0;
  	mesh_max_v = 0.0;
  	mesh_min_h = 0.0;
  	mesh_max_h = 0.0;

  	for(unsigned int i = 0; i<m.NrQuadrilateralStrips(); i++){
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
  }

  void updateDisplay();

  void AssignCloseDisplayPointer(bool* cd) {
    disp->close_display = cd;
  }

  void AssignMutexPointer(std::mutex* mu){
    disp->read_mutex = mu;
  }

  void LockMutex() {
    	if (read_mutex)
    		read_mutex->lock();
  }

  void UnlockMutex() {
    	if (read_mutex)
    		read_mutex->unlock();
  }

private:

  static Display* disp;

  Display();
  ~Display();

  bool *close_display;
  std::mutex* read_mutex;

  int lastTime;
  int delta;
  float rotator;

  int num_frames;
  std::chrono::milliseconds start_time;

  double mesh_min_v;
	double mesh_max_v;
	double mesh_min_h;
	double mesh_max_h;

  vector<Ode2DSystem*> _systems;
};

}

#endif
