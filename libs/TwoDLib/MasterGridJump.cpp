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

#include "MasterGridJump.hpp"

 using namespace TwoDLib;

 MasterGridJump::MasterGridJump
 (
	Ode2DSystemGroup& sys,
  double cell_width
):
_sys(sys),
_dydt(sys._vec_mass.size(),0.),
_cell_width(cell_width)
 {
 }

 void MasterGridJump::CalculateStaticEfficiacies(vector<double>& efficacy_map) {
   	for (MPILib::Index i = 0; i < efficacy_map.size(); i++){
      unsigned int offset = (unsigned int)abs(efficacy_map[i]/_cell_width);
      double goes = (double)fabs(efficacy_map[i] / _cell_width) - offset;
      double stays = 1.0 - goes;

      int offset_1 = efficacy_map[i] > 0 ? -offset : offset;
      int offset_2 = efficacy_map[i] > 0 ? -(offset+1) : (offset+1);

      _stays.push_back(vector<double>(_dydt.size()));
      _goes.push_back(vector<double>(_dydt.size()));
      _offset1.push_back(vector<int>(_dydt.size()));
      _offset2.push_back(vector<int>(_dydt.size()));

      for (MPILib::Index j=0; j < _dydt.size(); j++){
        _stays[i][j] = stays;
        _goes[i][j] = goes;
        _offset1[i][j] = offset_1;
        _offset2[i][j] = offset_2;
      }
    }
 }

 void MasterGridJump::CalculateStaticEfficiaciesForConductance(vector<double>& efficacy_map, vector<double>& rest_v) {
   	for (MPILib::Index i = 0; i < efficacy_map.size(); i++){
      _stays.push_back(vector<double>(_dydt.size()));
      _goes.push_back(vector<double>(_dydt.size()));
      _offset1.push_back(vector<int>(_dydt.size()));
      _offset2.push_back(vector<int>(_dydt.size()));

      for (MPILib::Index j=0; j < _dydt.size(); j++){
        double eff = efficacy_map[i] * (_sys.Vs()[j] - rest_v[i]);
        unsigned int offset = (unsigned int)abs(eff/_cell_width);
        double goes = (double)fabs(eff / _cell_width) - offset;
        double stays = 1.0 - goes;

        int offset_1 = eff > 0 ? -offset : offset;
        int offset_2 = eff > 0 ? -(offset+1) : (offset+1);

        _stays[i][j] = stays;
        _goes[i][j] = goes;
        _offset1[i][j] = offset_1;
        _offset2[i][j] = offset_2;
      }
    }
 }

 void MasterGridJump::MVGrid
 (
 	vector<double>&       dydt,
 	const vector<double>& vec_mass,
 	double                rate,
  unsigned int          efficiacy_index
 ) const
 {
 #pragma omp parallel for
 	for (MPILib::Index i = 0; i < dydt.size(); i++){
 		 dydt[i] += rate*_stays[efficiacy_index][i]*vec_mass[(((int)i+_offset1[efficiacy_index][i])%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
 		 dydt[i] += rate*_goes[efficiacy_index][i]*vec_mass[(((int)i+_offset2[efficiacy_index][i])%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
     dydt[i] -= rate*vec_mass[i];
 	}
 }

 void MasterGridJump::Apply(double t_step, const vector<double>& rates) {
	 _p_vec_rates = &rates;

	 typedef boost::numeric::odeint::runge_kutta_cash_karp54< vector<double> > error_stepper_type;
	 typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
	 controlled_stepper_type controlled_stepper;

	 boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ),
			 *this , _sys._vec_mass , 0.0 , t_step , 1e-4 );
 }

void MasterGridJump::operator()(const vector<double>& vec_mass, vector<double>& dydt, const double)
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
