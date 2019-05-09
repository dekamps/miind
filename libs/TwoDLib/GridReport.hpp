#ifndef _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD

#include <string>
#include "DensityAlgorithmInterface.hpp"

namespace TwoDLib {

template<class WeightValue>
class GridReport {

public:

  GridReport():
  _obs(std::map<MPILib::NodeId, DensityAlgorithmInterface<WeightValue>*>()) {

  }

  static GridReport<WeightValue>* getInstance() {
    if (!reg) {
      reg = new GridReport<WeightValue>();
    }

    return reg;
  }

  void registerObject(MPILib::NodeId id, DensityAlgorithmInterface<WeightValue>* obj) {
    _obs.insert(std::pair<MPILib::NodeId, DensityAlgorithmInterface<WeightValue>*>(id,obj));
  }

  void reportDensity(const std::vector<MPILib::NodeId>& node_ids, const std::vector<MPILib::Time>& start_times,
    const std::vector<MPILib::Time>& end_times, const std::vector<MPILib::Time>& intervals,  MPILib::Time time ) const {
    for (int i=0; i<node_ids.size(); i++){
      if ( _obs.find(node_ids[i]) == _obs.end() )
        continue;
      if(time >= start_times[i] && time <= end_times[i] && std::fabs(std::remainder(time, intervals[i])) < 0.00000001 ){
        _obs.at(node_ids[i])->reportDensity();
      }
    }
  }


private:
  std::map<MPILib::NodeId, DensityAlgorithmInterface<WeightValue>*> _obs;

  static GridReport<WeightValue>* reg;

};

template <class WeightValue>
GridReport<WeightValue>* GridReport<WeightValue>::reg;

}

#endif
