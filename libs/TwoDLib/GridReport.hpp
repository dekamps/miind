#ifndef _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD
#define _CODE_LIBS_TWODLIB_GRIDREPORT_INCLUDE_GUARD

#include <string>
#include "GridAlgorithm.hpp"

namespace TwoDLib {

template <class Algorithm>
class GridReport {

public:

  GridReport():
  _obs(map<MPILib::NodeId, Algorithm*>()) {

  }

  static GridReport<Algorithm>* getInstance() {
    if (!reg) {
      reg = new GridReport<Algorithm>();
    }

    return reg;
  }

  void registerObject(MPILib::NodeId id, Algorithm* obj) {
    _obs.insert(std::pair<MPILib::NodeId, Algorithm*>(id,obj));
  }

  void reportDensity(const vector<MPILib::NodeId>& node_ids, const vector<MPILib::Time>& start_times,
    const vector<MPILib::Time>& end_times, const vector<MPILib::Time>& intervals,  MPILib::Time time ) const {
    for (int i=0; i<node_ids.size(); i++){
      if ( _obs.find(node_ids[i]) == _obs.end() )
        continue;
      if(time >= start_times[i] && time <= end_times[i] && (time % intervals[i]) == 0 ){
        _obs[node_ids[i]]->reportDensity();
      }
    }
  }


private:
  map<MPILib::NodeId, Algorithm*> _obs;

  static GridReport<Algorithm>* reg;

};

template <class Algorithm>
GridReport<Algorithm>* GridReport<Algorithm>::reg;

}

#endif
