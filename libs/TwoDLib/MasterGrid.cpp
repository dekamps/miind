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
#include <random>

#include "MasterGrid.hpp"

 using namespace TwoDLib;

 MasterGrid::MasterGrid
 (
	Ode2DSystemGroup& sys,
  double cell_width
):
_sys(sys),
_dydt(sys._vec_mass.size(),0.),
_cell_width(cell_width)
 {
 }

 void MasterGrid::MVGrid
 (
 	vector<double>&       dydt,
 	const vector<double>& vec_mass,
 	double                rate,
   double stays,
   double goes,
   int offset_1,
   int offset_2
 ) const
 {
 #pragma omp parallel for
 	for (int i = 0; i < dydt.size(); i++){
 		 dydt[i] += rate*stays*vec_mass[(((int)i+offset_1)%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
 		 dydt[i] += rate*goes*vec_mass[(((int)i+offset_2)%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
     dydt[i] -= rate*vec_mass[i];
 	}
 }

 MPILib::Index MasterGrid::MVGridObject
 (
     MPILib::Index start_index, 
     int spikes,
     double stays,
     double goes,
     int offset_1,
     int offset_2
 ) const
 {
     MPILib::Index current_index = start_index;
     int max = (int)_sys._vec_cells_to_objects.size();

     for (int s = 0; s < spikes; s++) {
         double r1 = ((double)rand() / (double)RAND_MAX);
         if (r1 < stays) {
             current_index = (((int)current_index + offset_1) % max + max) % max;
         }
         else {
             current_index = (((int)current_index + offset_2) % max + max) % max;
         }
     }

     return current_index;
 }

 void MasterGrid::CalculateStaticEfficiacies(vector<double>& efficacy_map) {
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

 void MasterGrid::CalculateStaticEfficiaciesForConductance(vector<double>& efficacy_map, vector<double>& rest_v) {
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

 void MasterGrid::MVGridWithEfficacy
 (
 	vector<double>&       dydt,
 	const vector<double>& vec_mass,
 	double                rate,
  unsigned int          efficiacy_index
 ) const
 {
 #pragma omp parallel for
 	for (int i = 0; i < dydt.size(); i++){
 		 dydt[i] += rate*_stays[efficiacy_index][i]*vec_mass[(((int)i+_offset1[efficiacy_index][i])%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
 		 dydt[i] += rate*_goes[efficiacy_index][i]*vec_mass[(((int)i+_offset2[efficiacy_index][i])%(int)dydt.size()+(int)dydt.size()) % (int)dydt.size()];
     dydt[i] -= rate*vec_mass[i];
 	}
 }

 void MasterGrid::Apply(double t_step, const vector<double>& rates, vector<double>& efficacy_map) {
   _p_vec_eff   = &efficacy_map;
	 _p_vec_rates = &rates;

	 typedef boost::numeric::odeint::runge_kutta_cash_karp54< vector<double> > error_stepper_type;
	 typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
	 controlled_stepper_type controlled_stepper;

	 boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ),
			 *this , _sys._vec_mass , 0.0 , t_step , 1e-4 );
 }

 void MasterGrid::ApplyFinitePoisson
 (
     double t_step, 
     const vector<double>& rates, 
     vector<double>& efficacy_map
 )
 {
     static std::random_device rd;
     static std::mt19937 gen(rd());

#pragma omp parallel for
     for (int id = 0; id < _sys._vec_objects_to_index.size(); id++) {
         if (_sys._vec_objects_refract_times[id] >= 0.0)
             continue;

         for (unsigned int irate = 0; irate < rates.size(); irate++) {
            if (rates[irate] == 0)
                continue;

            unsigned int offset = (unsigned int)abs(efficacy_map[irate] / _cell_width);
            double goes = (double)fabs(efficacy_map[irate] / _cell_width) - offset;
            double stays = 1.0 - goes;

            int offset_1 = efficacy_map[irate] > 0 ? offset : -offset;
            int offset_2 = efficacy_map[irate] > 0 ? (offset + 1) : -(offset + 1);

            std::poisson_distribution<int> pd(rates[irate] * t_step);
            int spikes = pd(gen);
            _sys._vec_objects_to_index[id] = MVGridObject(_sys._vec_objects_to_index[id], spikes, stays, goes, offset_1, offset_2);
         }
     }

     _sys.updateVecCellsToObjects();
 }

 void MasterGrid::operator()(const vector<double>& vec_mass, vector<double>& dydt, const double)
{
  const vector<double>& vec_eff = *_p_vec_eff;
  const vector<double>& rates = *_p_vec_rates;


#pragma omp parallel for
 for(int id = 0; id < dydt.size(); id++)
   dydt[id] = 0.;

 for (unsigned int irate = 0; irate < rates.size(); irate++){
   // do NOT map the rate
    double rate = rates[irate];

    unsigned int offset = (unsigned int)abs(vec_eff[irate]/_cell_width);
    double goes = (double)fabs(vec_eff[irate] / _cell_width) - offset;
    double stays = 1.0 - goes;

    int offset_1 = vec_eff[irate] > 0 ? -offset : offset;
    int offset_2 = vec_eff[irate] > 0 ? -(offset+1) : (offset+1);

    // it is only the matrices that need to be mapped
    MVGrid
   (
       dydt,
     vec_mass,
     rate,
     stays,
     goes,
     offset_1,
     offset_2
    );

    // MVGridWithEfficacy
    // (
    //   dydt,
    // vec_mass,
    // rate,
    // irate
    // );
  }
}
