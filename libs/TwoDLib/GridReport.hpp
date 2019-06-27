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

  void reportDensity(const std::vector<MPILib::NodeId>& node_ids, std::vector<MPILib::Time>& start_times,
    const std::vector<MPILib::Time>& end_times, const std::vector<MPILib::Time>& intervals,  MPILib::Time time ) const {
    for (int i=0; i<node_ids.size(); i++){
      if ( _obs.find(node_ids[i]) == _obs.end() )
        continue;
      // Bit of a nasty hack here : using start_times to track the current time.
      if(time >= start_times[i]+intervals[i] && time <= end_times[i]){
        start_times[i] += intervals[i];
        _obs.at(node_ids[i])->reportDensity(start_times[i]);
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
