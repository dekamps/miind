// Copyright (c) 2005 - 2015 Marc de Kamps
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
//
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <numeric>

#include "MasterGridSomaDendrite.hpp"

 using namespace TwoDLib;

 MasterGridSomaDendrite::MasterGridSomaDendrite
 (
	Ode2DSystemGroup& sys,
  double cell_width
):
_sys(sys),
_dydt(sys._vec_mass.size(),0.),
_cell_width(cell_width)
 {
 }

 void MasterGridSomaDendrite::InitializeEfficacyVectors(unsigned int size){
   for (MPILib::Index i = 0; i < size; i++){
     _stays.push_back(std::map<int, vector<double>>());
     _goes.push_back(std::map<int, vector<double>>());
   }
 }

 void MasterGridSomaDendrite::CalculateDynamicEfficiacies(vector<std::string>& conn_types, vector<double>& efficacy_map, vector<double>& rest_v, vector<double>& conductances) {
#pragma omp parallel for
   	for (MPILib::Index i = 0; i < efficacy_map.size(); i++){
      std::map<int,vector<double>>::iterator iter = _stays[i].begin();
#pragma omp parallel for
      for (int n = 0; n<_stays[i].size(); n++){
        #pragma omp parallel for
        for (MPILib::Index j=0; j < _dydt.size(); j++){
          iter->second[j] = 0.0;
        }
       iter++;
      }
      iter = _goes[i].begin();
#pragma omp parallel for
      for (int n = 0; n<_goes[i].size(); n++){
        #pragma omp parallel for
        for (MPILib::Index j=0; j < _dydt.size(); j++){
          iter->second[j] = 0.0;
        }
       iter++;
      }

      for (MPILib::Index j=0; j < _dydt.size(); j++){
        double eff = efficacy_map[i];
        if (conn_types[i] == "SomaDendrite")
          eff = 0.0001 * conductances[i] * (rest_v[i] - _sys.Vs()[j]);
        unsigned int offset = (unsigned int)abs(eff/_cell_width);
        double goes = (double)fabs(eff / _cell_width) - offset;
        double stays = 1.0 - goes;

        int offset_1 = eff > 0 ? -offset : offset;
        int offset_2 = eff > 0 ? -(offset+1) : (offset+1);

        if(_stays[i].find(offset_1) == _stays[i].end()){
          _stays[i][offset_1] = vector<double>(_dydt.size());
        }

        if(_goes[i].find(offset_2) == _goes[i].end()){
          _goes[i][offset_2] = vector<double>(_dydt.size());
        }

        _stays[i][offset_1][(((int)j-offset_1)%(int)_dydt.size()+(int)_dydt.size()) % (int)_dydt.size()] = stays;
        _goes[i][offset_2][(((int)j-offset_2)%(int)_dydt.size()+(int)_dydt.size()) % (int)_dydt.size()] = goes;
      }
    }
 }

 void MasterGridSomaDendrite::MVGrid
 (
 	vector<double>&       dydt,
 	const vector<double>& vec_mass,
 	double                rate,
  unsigned int          efficiacy_index
 ) const
 {
  for (std::map<int,vector<double>>::const_iterator iter = _stays[efficiacy_index].begin(); iter != _stays[efficiacy_index].end(); ++iter){
#pragma omp parallel for
    for (MPILib::Index i = 0; i < iter->second.size(); i++){
      dydt[i] += rate*iter->second[i]*vec_mass[(((int)i+iter->first)%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
   	}
  }
  for (std::map<int,vector<double>>::const_iterator iter = _goes[efficiacy_index].begin(); iter != _goes[efficiacy_index].end(); ++iter){
#pragma omp parallel for
    for (MPILib::Index i = 0; i < iter->second.size(); i++){
      dydt[i] += rate*iter->second[i]*vec_mass[(((int)i+iter->first)%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
    }
  }
#pragma omp parallel for
 	for (MPILib::Index i = 0; i < dydt.size(); i++){
    dydt[i] -= rate*vec_mass[i];
 	}
 }

 void MasterGridSomaDendrite::Apply(double t_step, const vector<double>& rates) {
	 _p_vec_rates = &rates;

	 typedef boost::numeric::odeint::runge_kutta_cash_karp54< vector<double> > error_stepper_type;
	 typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
	 controlled_stepper_type controlled_stepper;

	 boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ),
			 *this , _sys._vec_mass , 0.0 , t_step , 1e-4 );
 }

void MasterGridSomaDendrite::operator()(const vector<double>& vec_mass, vector<double>& dydt, const double)
{
  const vector<double>& rates = *_p_vec_rates;


#pragma omp parallel for
  for(unsigned int id = 0; id < dydt.size(); id++)
    dydt[id] = 0.;

  for (unsigned int irate = 0; irate < rates.size(); irate++){
    double rate = rates[irate];

    MVGrid
    (
      dydt,
      vec_mass,
      rate,
      irate
    );
  }
}
