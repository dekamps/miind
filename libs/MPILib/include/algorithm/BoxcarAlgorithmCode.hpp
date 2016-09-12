// Copyright (c) 2005 - 2012 Marc de Kamps
//                      2012 David-Matthias Sichau
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation
//      and/or other materials provided with the distribution.
//    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software
//      without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <MPILib/include/utilities/ParallelException.hpp>
#include <MPILib/include/TypeDefinitions.hpp>
#include <MPILib/include/algorithm/BoxcarAlgorithm.hpp>
#include <MPILib/include/StringDefinitions.hpp>
#include <MPILib/include/BasicDefinitions.hpp>

#include <NumtoolsLib/NumtoolsLib.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>

#include <functional>
#include <numeric>

namespace MPILib {
namespace algorithm {

template<class Weight>
BoxcarAlgorithm<Weight>::BoxcarAlgorithm(std::vector<Event>& events, Rate change) :
        AlgorithmInterface<Weight>(), _time_current(
                std::numeric_limits<double>::max()), _rate(0.0),
                _events(events), _n_events(events.size()), _current_event(0),
                _event_on(false), _change_factor(change), _counter(0){}

template<class Weight>
BoxcarAlgorithm<Weight>::~BoxcarAlgorithm() {}

template<class Weight>
BoxcarAlgorithm<Weight>* BoxcarAlgorithm<Weight>::clone() const {
    return new BoxcarAlgorithm(*this);
}

template<class Weight>
void BoxcarAlgorithm<Weight>::configure(const SimulationRunParameter& simParam) {
    _time_current = simParam.getTBegin();
}

template<class Weight>
void BoxcarAlgorithm<Weight>::evolveNodeState(const std::vector<Rate>& nodeVector,
        const std::vector<Weight>& weightVector, Time time) {

    try
    {
        if(weightVector.size() > 0)
        {
            cout << weightVector.size() << endl;
            cout << nodeVector.size() << endl;
            string s = "no connections should be done to a BoxcarNode";
            throw s;
        };
    }
    catch(string s)
    {
        cout << s ; throw 0;
    }

    _time_current = time;

    if(_current_event < _n_events)
    {
        if (_event_on){
            if (_time_current >= _events[_current_event].end){
                // cout << "event ended";
                _event_on = false;
                _current_event ++;
            }
            else{
                if (_rate < _events[_current_event].rate){
                    _rate += _change_factor;
                }; // Stuck in rate if next event is soon after and lower
            };
        }
        else {
            if (_time_current >= _events[_current_event].start){
                // cout << " event started ";
                _event_on = true;
                // _change_factor = _events[_current_event].rate/10000.0
            }
            else{
                if (_rate - _change_factor > 0.0){
                    _rate -= _change_factor;
                }
                else {
                    _rate = 0.0;
                };
            };
        };
    }
    else{
        if ((_rate - _change_factor) > 0.0){
            _rate -= _change_factor;
        }
        else {
            _rate = 0.0;
        };
    };

}

template<class Weight>
Time BoxcarAlgorithm<Weight>::getCurrentTime() const {
    return _time_current;
}

template<class Weight>
Rate BoxcarAlgorithm<Weight>::getCurrentRate() const {
    return _rate;
}

template<class Weight>
AlgorithmGrid BoxcarAlgorithm<Weight>::getGrid() const {
    std::vector<double> array_state(1);
    array_state[0] = _rate;
    AlgorithmGrid grid(array_state);
    return grid;
}

} /* namespace algorithm */
} /* namespace MPILib */
