#include "NdPoint.hpp"

NdPoint::NdPoint(std::vector<double> _coords) :
    coords(_coords),
    lives(0),
    dead(false),
    hyper(false) {}

NdPoint::NdPoint() :
    coords(0),
    lives(0),
    dead(false),
    hyper(false) {}

NdPoint& NdPoint::operator=(const NdPoint& other) {
    coords = std::vector<double>(other.coords.size());
    for (unsigned int i = 0; i < other.coords.size(); i++)
        coords[i] = other.coords[i];

    lives = other.lives;
    dead = other.dead;
    hyper = other.hyper;

    return *this;
}

bool NdPoint::operator==(const NdPoint& other) const {
    if (other.coords.size() != coords.size())
        return false;

    bool equal = true;
    for (unsigned int i = 0; i < coords.size(); i++) {
        equal &= (coords[i] == other.coords[i]);
    }
    return equal;
}