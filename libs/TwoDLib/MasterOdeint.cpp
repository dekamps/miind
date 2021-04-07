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

#include "MasterOdeint.hpp"


 using namespace TwoDLib;

 MasterOdeint::MasterOdeint
 (
	Ode2DSystemGroup& sys,
	const std::vector<vector<TransitionMatrix> >& vec_vec_mat,
	const MasterParameter& par
):
_sys(sys),
_vec_vec_mat(vec_vec_mat),
_vec_vec_csr(InitializeCSR(vec_vec_mat,sys)),
_par(par),
_dydt(sys._vec_mass.size(),0.),
_rate(0.0),
_p_vec_map(0),
_p_vec_rates(0)
 {
 }

MasterOdeint::MasterOdeint
(
	const MasterOdeint& rhs
):
_sys(rhs._sys),
_vec_vec_mat(rhs._vec_vec_mat),
_vec_vec_csr(rhs._vec_vec_csr),
_par(rhs._par),
_dydt(_sys._vec_mass.size(),0.),
_p_vec_map(rhs._p_vec_map),    // this is not pretty, but MasterOdeint owns neither the sys objects, nor the map object
_p_vec_rates(rhs._p_vec_rates) // nor the rates object. integrate and variations spawn copies. they need to refer to the same objects.
{
}

 void MasterOdeint::Apply
 (
	double t_step,
	const std::vector< std::vector<double> >& rates,
	const vector<MPILib::Index>& vec_map
)
 {
	 _p_vec_map   = &vec_map;
	 _p_vec_rates = &rates;

	 typedef boost::numeric::odeint::runge_kutta_cash_karp54< vector<double>, double, vector<double>, double> error_stepper_type;
	 typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
	 controlled_stepper_type controlled_stepper;

	 boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled< error_stepper_type >( 1.0e-6 , 1.0e-6 ),
			 std::ref(*this) , _sys._vec_mass , 0.0 , t_step , 1e-4 );

 }

 void MasterOdeint::ApplyFinitePoisson
 (
	 double t_step,
	 const std::vector< std::vector<double > >& vec_vec_rates,
	 const vector<MPILib::Index>& vec_vec_map
 )
 {
	 static std::random_device rd;
	 static std::mt19937 gen(rd());
	 MPILib::Number n_mesh = vec_vec_rates.size();

#pragma omp parallel for
	 for (int id = 0; id < _sys._vec_objects_to_index.size(); id++) {
		 if (_sys._vec_objects_refract_times[id] >= 0.0)
			 continue;

		 for (MPILib::Index mesh_index = 0; mesh_index < n_mesh; mesh_index++) {
			 for (unsigned int irate = 0; irate < vec_vec_rates[mesh_index].size(); irate++) {
				 if (vec_vec_rates[mesh_index][irate] == 0)
					 continue;

				 std::poisson_distribution<int> pd(vec_vec_rates[mesh_index][irate] * t_step);
				 int spikes = pd(gen);
				 _sys._vec_objects_to_index[id] =
					 _vec_vec_csr[mesh_index][vec_vec_map[irate]].MVObject(_sys._vec_objects_to_index[id], spikes);
			 }
		 }
	 }

	 _sys.updateVecCellsToObjects();
 }

std::vector<std::vector<CSRMatrix> > MasterOdeint::InitializeCSR(const std::vector< std::vector<TransitionMatrix> >& vec_vec_mat, const Ode2DSystemGroup& sys)
 {
	 std::vector<std::vector<CSRMatrix> > vec_ret;
	 MPILib::Index mesh_index = 0;
	 for (const auto& vec_mat: vec_vec_mat){
		 std::vector<CSRMatrix> vec_dummy;
		 for (const auto& mat: vec_mat)
			 vec_dummy.push_back(CSRMatrix(mat,sys,mesh_index));
		 vec_ret.push_back(vec_dummy);
		 mesh_index++;
	 }
	 return vec_ret;
 }

 void MasterOdeint::operator()(const vector<double>& vec_mass, vector<double>& dydt, const double)
{
	 const std::vector<MPILib::Index>& vec_map            = *_p_vec_map;
	 const std::vector<std::vector<MPILib::Rate> >& rates = *_p_vec_rates;


#pragma omp parallel for
	for(int id = 0; id < dydt.size(); id++)
		dydt[id] = 0.;

	MPILib::Number nr_meshes = _p_vec_rates->size();
	for (MPILib::Index i_mesh = 0; i_mesh < nr_meshes; i_mesh++){
		for (unsigned int irate = 0; irate < rates[i_mesh].size(); irate++){
		// do NOT map the rate
			_rate = rates[i_mesh][irate];
			// it is only the matrices that need to be mapped
			_vec_vec_csr[i_mesh][vec_map[irate]].MVMapped
			(
				dydt,
				vec_mass,
				_rate
			);
		}
	}
}
