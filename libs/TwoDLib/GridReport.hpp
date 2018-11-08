#ifndef _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD

#include <string>
#include "GridAlgorithm.hpp"

namespace TwoDLib {

template <class WeightValue>
class GridReport {

public:

  GridReport():
  _obs(vector<GridAlgorithm<WeightValue>*>()) {

  }

  static GridReport<WeightValue>* getInstance() {
    if (!reg) {
      reg = new GridReport<WeightValue>();
    }

    return reg;
  }

  void registerObject(GridAlgorithm<WeightValue>* obj) {
    _obs.push_back(obj);
  }

  void reportFiringRate() const {
    for (int i=0; i<_obs.size(); i++)
      _obs[i]->reportFiringRate();
  }

  void reportDensity() const {
    for (int i=0; i<_obs.size(); i++)
      _obs[i]->reportDensity();
  }


private:
  vector<GridAlgorithm<WeightValue>*> _obs;

  static GridReport<WeightValue>* reg;

};

template <class WeightValue>
GridReport<WeightValue>* GridReport<WeightValue>::reg;

}

#endif
