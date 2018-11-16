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

#include "MasterGrid.hpp"

 using namespace TwoDLib;

 MasterGrid::MasterGrid
 (
	Ode2DSystem& sys,
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
 	for (MPILib::Index i = offset_2; i < dydt.size()-(offset_2); i++){
 		dydt[i] += rate*stays*vec_mass[i+offset_1];
 		dydt[i] += rate*goes*vec_mass[i+offset_2];
 	  dydt[i] -= rate*vec_mass[i];
 	}
 }

 void MasterGrid::MVGridMapped
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
 	for (MPILib::Index i = 0; i < _sys.MeshObject().NrQuadrilateralStrips(); i++){
    // When mapping, we have to ensure that the ends of the strip don't remove mass
    // as there's nowhere for it to go
    MPILib::Index i_r =_sys.Map(i,std::max(0,-offset_2));
    dydt[i_r] += rate*stays*vec_mass[i_r+offset_1];
    dydt[i_r] += rate*goes*vec_mass[i_r+offset_2];
    for (MPILib::Index j = std::max(1,1-offset_2); j < std::min(_sys.MeshObject().NrCellsInStrip(i)-1-offset_2, _sys.MeshObject().NrCellsInStrip(i)-1); j++){
      MPILib::Index i_r =_sys.Map(i,j);
      dydt[i_r] += rate*stays*vec_mass[i_r+offset_1];
      dydt[i_r] += rate*goes*vec_mass[i_r+offset_2];
      dydt[i_r] -= rate*vec_mass[i_r];
    }
    i_r =_sys.Map(i,std::min(_sys.MeshObject().NrCellsInStrip(i)-1-offset_2, _sys.MeshObject().NrCellsInStrip(i)-1));
    dydt[i_r] += rate*stays*vec_mass[i_r+offset_1];
    dydt[i_r] += rate*goes*vec_mass[i_r+offset_2];
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

 void MasterGrid::operator()(const vector<double>& vec_mass, vector<double>& dydt, const double)
{
  const vector<double>& vec_eff = *_p_vec_eff;
  const vector<double>& rates = *_p_vec_rates;


#pragma omp parallel for
 for(unsigned int id = 0; id < dydt.size(); id++)
   dydt[id] = 0.;

 for (unsigned int irate = 0; irate < rates.size(); irate++){
   // do NOT map the rate
    double rate = rates[irate];

    unsigned int offset = (unsigned int)abs(vec_eff[irate]/_cell_width);
    double goes = fabs(vec_eff[irate] / _cell_width) - offset;
    double stays = 1.0 - goes;

    int offset_1 = vec_eff[irate] > 0 ? -offset : offset;
    int offset_2 = vec_eff[irate] > 0 ? -(offset+1) : -(offset-1);

    // it is only the matrices that need to be mapped
    MVGridMapped
   (
       dydt,
     vec_mass,
     rate,
     stays,
     goes,
     offset_1,
     offset_2
    );
  }
}
