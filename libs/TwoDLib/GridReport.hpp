#ifndef _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD

#include <string>
#include <MPILib/include/ReportRegister.hpp>
#include "GridAlgorithm.hpp"

namespace TwoDLib {

template <class WeightValue>
class GridReport {

public:

  static ReportRegister<GridAlgorithm<WeightValue>>* getInstance() {
    if (!reg) {
      reg = new GridReport<WeightValue>();
    }

    return reg;
  }

  void reportFiringRate() {
    for (int i=0; i<ReportRegister<GridAlgorithm<WeightValue>>::_obs.size(); i++)
      ReportRegister<GridAlgorithm<WeightValue>>::_obs[i]._system->getGrid(i, true);
  }

  void reportDensity() {
    for (int i=0; i<ReportRegister<GridAlgorithm<WeightValue>>::_obs.size(); i++)
      ReportRegister<GridAlgorithm<WeightValue>>::_obs[i]._system->getGrid(i, true);
  }


private:
  _obs = vector<ReportObject<System>>();

  static ReportRegister<GridAlgorithm<WeightValue>>* reg;

};

template <class WeightValue>
ReportRegister<GridAlgorithm<WeightValue>>* GridReport<WeightValue>::reg;

}

#endif
