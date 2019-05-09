#ifndef TWODLIB_DENSITYALGORITHMINTERFACE_HPP_
#define TWODLIB_DENSITYALGORITHMINTERFACE_HPP_

#include <MPILib/include/AlgorithmInterface.hpp>

namespace TwoDLib {

template<class WeightValue>
class DensityAlgorithmInterface : public MPILib::AlgorithmInterface<WeightValue> {
public:

  virtual void reportDensity() const = 0;

};

}
#endif
