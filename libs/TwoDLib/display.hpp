#ifndef _CODE_LIBS_TWODLIB_DISPLAY_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_DISPLAY_INCLUDE_GUARD

#define LINUX // This #define is used in glew.c so it does not require installation
#include "include/GL/glew.h"
// The application does require glut to be installed
#include <GL/freeglut.h>

#include <string>
#include <chrono>
#include "Ode2DSystemGroup.hpp"
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/DelayedConnection.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <mutex>
#include <map>

typedef MPILib::MPINetwork<MPILib::DelayedConnection, MPILib::utilities::CircularDistribution> Network;

namespace TwoDLib {

class DisplayWindow{
public:

  Ode2DSystemGroup* _system;
  int _window_index;

  double mesh_min_v;
  double mesh_max_v;
  double mesh_min_h;
  double mesh_max_h;
};

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
  void animate(bool,std::vector<MPILib::NodeId>, double time_step) const;
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

  unsigned int addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys);

  void updateDisplay(long current_sim_it);

private:

  bool write_frames;
  void writeFrame(unsigned int system, long frame_num);

  static Display* disp;

  Display();
  ~Display();

  long _current_sim_it;
  double _time_step;

  int lastTime;

  std::vector<MPILib::NodeId> _nodes_to_display;

  std::chrono::milliseconds start_time;

  std::map<MPILib::NodeId, DisplayWindow> _dws;
};

}

#endif
