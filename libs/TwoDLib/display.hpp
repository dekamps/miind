#ifndef _CODE_LIBS_TWODLIB_DISPLAY_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_DISPLAY_INCLUDE_GUARD

#define LINUX // This #define is used in glew.c so it does not require installation
#include "include/GL/glew.h"
#ifndef USING_APPLE_GLUT
// The application does require glut to be installed
#include <GL/freeglut.h>
#else
#include <GLUT/glut.h>
#endif

#include <string>
#include <chrono>
#include "Ode2DSystemGroup.hpp"
#include <MPILib/include/MPINetwork.hpp>
#include <MPILib/include/MPINetworkCode.hpp>
#include <MPILib/include/DelayedConnection.hpp>
#include <MPILib/include/utilities/CircularDistribution.hpp>
#include <mutex>
#include <map>

namespace TwoDLib {

class DisplayWindow{
public:

  Ode2DSystemGroup* _system;
  unsigned int _mesh_index;
  int _window_index;
  bool _3D;

  double rot_x;
  double rot_y;
  int dim_select;

  double mesh_min_v;
  double mesh_max_v;
  double mesh_min_h;
  double mesh_max_h;

  double max_mass;
  double min_mass;

  double width;
  double height;
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
  void setDisplayNodes(std::vector<MPILib::NodeId> nodes_to_display) const ;
  void animate(bool, double time_step) const;
  void processDraw(void);

  void display_3d(void);
  void scene_3d(int width, int height);
  void init_3d() const;
  void update_3d();
  void keyboard_3d_down(int key, int _x, int _y);
  void keyboard_3d_up(int key, int _x, int _y);

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

  static void stat_display_3d(void) {
      disp->display_3d();
  }
  static void stat_scene_3d(int width, int height) {
      disp->scene_3d(width, height);
  }
  static void stat_update_3d(void) {
      disp->update_3d();
  }

  static void stat_keyboard_3d_down(int key, int _x, int _y) {
      disp->keyboard_3d_down(key, _x, _y);
  }

  static void stat_keyboard_3d_up(int key, int _x, int _y) {
      disp->keyboard_3d_up(key, _x, _y);
  }

  unsigned int addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys, bool _3d, unsigned int mesh_index);
  unsigned int addOdeSystem(MPILib::NodeId nid, Ode2DSystemGroup* sys, bool _3d);

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

  bool upPressed;
  bool downPressed;
  bool leftPressed;
  bool rightPressed;
  bool pgupPressed;
  bool pgdnPressed;
};

}

#endif
