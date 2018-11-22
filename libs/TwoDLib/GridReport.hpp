#ifndef _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD

#include <string>
#include "GridAlgorithm.hpp"

namespace TwoDLib {

template <class Algorithm>
class GridReport {

public:

  GridReport():
  _obs(vector<Algorithm*>()) {

  }

  static GridReport<Algorithm>* getInstance() {
    if (!reg) {
      reg = new GridReport<Algorithm>();
    }

    return reg;
  }

  void registerObject(Algorithm* obj) {
    _obs.push_back(obj);
  }

  void reportDensity() const {
    for (int i=0; i<_obs.size(); i++)
      _obs[i]->reportDensity();
  }


private:
  vector<Algorithm*> _obs;

  static GridReport<Algorithm>* reg;

};

template <class Algorithm>
GridReport<Algorithm>* GridReport<Algorithm>::reg;

}

#endif
