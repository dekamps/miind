#ifndef APP_ND_GRID_POINT
#define APP_ND_GRID_POINT

#include <vector>

class NdPoint {
public:
    std::vector<double> coords;
    int lives;
    bool dead;
    bool hyper;

    NdPoint(std::vector<double> _coords);
    NdPoint();
    NdPoint& operator=(const NdPoint& other);
    bool operator==(const NdPoint& other) const;
};

#endif