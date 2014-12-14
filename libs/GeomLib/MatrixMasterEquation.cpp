// Copyright (c) 2005 - 2013 Marc de Kamps
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

//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net
#include "MatrixMasterEquation.h"

using namespace GeomLib;

MatrixMasterEquation::MatrixMasterEquation
(
     SpikingOdeSystem& 	system,
     Number	       	n_iter,
     Index	      	i_reset,	
     const Precision&  	precision,	
     Time	      	t_begin,
     Time	       	t_step		
 ):
 _system	      	(system),
 _vec_potential		(system.InterpretationBuffer()),
 _vec_density		(system.MassBuffer()),
 _i_reset	       	(system.IndexResetBin()),
 _vec_matrix_state	(system.NumberOfBins()),
 _matrix_integrator
 (
     n_iter,
     &_vec_matrix_state[0],
     _vec_matrix_state.size(),
     t_step,
     t_begin,
     precision,
     GeomLib::DerivMatrixVersion
 ),
 _mat_transit(system.NumberOfBins(), system.NumberOfBins())
 {
       	_mat_transit.SetZero();
 }

 void MatrixMasterEquation::Initialize
 (
       	Index				  index,
       	const vector<InputParameterSet>&  vec_set
 )
 {
       	//MdK: 14-02-2013: also copy the density state prior to evolution since this may have been
       	//changed by external agents, e.g. the refractory queue; the densities are then not in sync 
       	// any more and would trigger the comparison assert

       	std::copy(_vec_density.begin(),_vec_density.end(),_vec_matrix_state.begin());


       	_matrix_integrator.Parameter()._p_mat	    = &_mat_transit;
       	_matrix_integrator.Parameter()._nr_bins	    = _vec_density.size();
       	_matrix_integrator.Parameter()._p_vec_set	= &vec_set;
       	_matrix_integrator.Parameter()._p_system	= &_system;
       	_matrix_integrator.Parameter()._i_reset		= _i_reset;
       	_matrix_integrator.Parameter()._p_cache		= 0;
 }

	bool SumColumnsZero(const NumtoolsLib::QaDirty<double>& mat)
	{
			for (long i = 0; i < mat.NrXdim(); i++){
				double sum = 0.0;
				for(long j = 0; j < mat.NrYdim(); j++){
					sum += mat(j,i);
				}
				if (fabs(sum) > 1e-8)
					return false;
			}

			return true;
		}

	bool ProbabilityConservation(const double dydt[], Number n_bins){
		double sum = 0.0;
		for (Index i = 0; i < n_bins; i++)
			sum += dydt[i];
		return (fabs(sum) < 1e-8) ? true : false;
	}


		pair<Potential, Potential> BinLimits
			(
				Potential V_max, 
				const vector<Potential>& array_interpretation, 
				Index i
			)
		{			
			pair<Potential,Potential> pair_ret;

			assert ( i < array_interpretation.size() );

			Potential lower = array_interpretation[i];
			Potential upper = ( i < array_interpretation.size() - 1 ) ? array_interpretation[i+1] : V_max;
			pair_ret.first  = lower;
			pair_ret.second = upper;

			return pair_ret;
		}

		double LowerFraction
		(
			Potential V_max,
			const vector<double>& array_interpretation, 
			int i, 
			Potential v
		)
		{	
			if (i < 0 || i >= static_cast<int>(array_interpretation.size()) )
				return 0.0;

			pair<Potential,Potential> bin_limits = BinLimits(V_max,array_interpretation,i);
			assert( v >= bin_limits.first && v <= bin_limits.second); 
			double frac = (bin_limits.second - v)/(bin_limits.second - bin_limits.first);
			return frac;
		}


		double HigherFraction
		(
			Potential V_max,
			const vector<double>& array_interpretation, 
			int i, 
			Potential v
		)
		{
			if (i >= static_cast<int>(array_interpretation.size()) || i < 0 )
				return 0.0;

			pair<Potential,Potential> bin_limits = BinLimits(V_max,array_interpretation,i);
			assert( v >= bin_limits.first && v <= bin_limits.second); 
			double frac = (v - bin_limits.first)/(bin_limits.second - bin_limits.first);
			return frac;
		}

 void AddSameBin
 (
	Index		i,
    Index    	i_add,
    Potential  	lower,
   	Potential  	higher,
   	NumtoolsLib::QaDirty<Potential>&	mat,
   	const vector<Potential>&	       	array_interpretation,
   	Potential 	V_max,
   	Rate		rate
 )
 {    
        if (i_add >= array_interpretation.size()  )
	  return;

      	pair<Potential, Potential> bin_limits = BinLimits(V_max,array_interpretation,i_add);
       	assert( lower  >= bin_limits.first && lower  <= bin_limits.second );
       	assert( higher >= bin_limits.first && higher <= bin_limits.second );
       	double frac = (higher - lower)/(bin_limits.second - bin_limits.first);
       	mat(i,i_add) += rate*frac;

 }
		void AddCongruentBins
		(
			Index								i,
			int									i_lower,
			int									i_higher,
			Potential							lower,
			Potential							higher,
			NumtoolsLib::QaDirty<Potential>&	mat,
			const vector<Potential>&			array_interpretation,
			Potential							V_max,
			Rate								rate
		)
		{
			if ( i_lower >= 0){
				double frac_lower = LowerFraction(V_max, array_interpretation, i_lower, lower);
				mat(i,i_lower) += rate*frac_lower;
			}

			if ( i_higher < static_cast<int>(array_interpretation.size()) ){
				double frac_higher = HigherFraction(V_max, array_interpretation, i_higher, higher);
				mat(i,i_higher) += rate*frac_higher;
			}

			for (int j = i_lower + 1; j < i_higher; j++)
				mat(i,j) += rate;
		}

		void AddRow
		(
			Index							i,
			NumtoolsLib::QaDirty<double>&	mat,
			int								i_lower,
			double							lower,
			int								i_higher,
			double							higher,
			const vector<double>&			array_interpretation,
			double							V_max,
			double							rate
		)
		{
				if (i_lower == i_higher){
					AddSameBin(i,i_lower,lower,higher,mat,array_interpretation,V_max, rate);
					return;
				}
				if (i_lower < i_higher){
					AddCongruentBins(i, i_lower, i_higher, lower, higher, mat, array_interpretation, V_max, rate);
					return;
				}
		}

		int find_bin (Index i, const vector<double>& array_interpretation, Potential V_max, double tr){
			if (tr < array_interpretation[0] )
				return -1;
			if (tr > V_max)
				return array_interpretation.size();

			Index i_upper;
			if (tr< array_interpretation[i] ){
				i_upper = i;
				while (array_interpretation[--i_upper] > tr)
					;
			}
			else {
				i_upper = array_interpretation.size(); // this is not an array validation, see below
				while(array_interpretation[--i_upper] > tr )
					;
			}
			return i_upper;
		}


	void HandleExcitatoryInput
		(
			NumtoolsLib::QaDirty<double>&	mat,
			const InputParameterSet&		set,
			const vector<double>&			array_interpretation,
			Potential						V_max,
			Potential						V_reset,
			Index							i_reset
		)
	{
		Potential tr_z = V_max - set._h_exc;
		Index i_z = find_bin(0,array_interpretation, V_max, tr_z);
		double frac = LowerFraction(V_max, array_interpretation,i_z,tr_z);

		Number n_bins = array_interpretation.size();
		for (Index i = 0; i < n_bins; i++){
			pair<Potential,Potential> bin_limits = BinLimits(V_max,array_interpretation,i);

			double tr_u = bin_limits.second - set._h_exc;
			double tr_l = bin_limits.first - set._h_exc;

			// find bins
			int i_upper = find_bin(i, array_interpretation, V_max, tr_u);
			int i_lower = find_bin(i, array_interpretation, V_max, tr_l);

			mat(i,i) -= set._rate_exc;

			if (i == i_reset){
				mat(i_reset,i_z) += set._rate_exc*frac;
			    for (Index j= i_z + 1; j < n_bins; j++)
					mat(i_reset,j) += set._rate_exc;
			}

			AddRow(i,mat, i_lower, tr_l, i_upper, tr_u ,array_interpretation,V_max,set._rate_exc);
		}
	}


	void HandleInhibitoryInput
		(
			NumtoolsLib::QaDirty<double>&	mat,
			const InputParameterSet&		set,
			const vector<double>&			array_interpretation,
			Potential						V_max
		)
	{
		Potential tr_z = array_interpretation[0] - set._h_inh;
		Index i_z = find_bin(0,array_interpretation, V_max, tr_z);
		double frac_lower = LowerFraction(V_max, array_interpretation,i_z,tr_z);
		
		for (Index i = 0; i < array_interpretation.size(); i++){
			pair<Potential,Potential> bin_limits = BinLimits(V_max,array_interpretation,i);
			assert( set._h_inh < 0);

			double tr_u = bin_limits.second - set._h_inh;
			double tr_l = bin_limits.first - set._h_inh;
		
			// find bins
			int i_upper = find_bin(i, array_interpretation, V_max, tr_u);
			int i_lower = find_bin(i, array_interpretation, V_max, tr_l);

			if (i == i_z)
				mat(i,i) -= set._rate_inh*frac_lower;
			if (i > i_z)
				mat(i,i) -= set._rate_inh;

			AddRow(i,mat, i_lower, tr_l, i_upper, tr_u ,array_interpretation,V_max,set._rate_inh);
		}
	}


	void InitializeMatrix
	(
		NumtoolsLib::QaDirty<double>&		mat, 
		const vector<InputParameterSet>&	vec_set,
		const vector<double>&		       	array_interpretation,
		Potential      				V_max,
		Potential		      		V_reset,
		Index			       		i_reset
	)
	{
		mat.SetZero();

		for (Index i = 0; i < vec_set.size(); i++){
			if (vec_set[i]._rate_exc > 0 )
				HandleExcitatoryInput(mat,vec_set[i],array_interpretation,V_max, V_reset, i_reset);
			if (vec_set[i]._rate_inh > 0 )
				HandleInhibitoryInput(mat,vec_set[i],array_interpretation,V_max);
		}

		assert(SumColumnsZero(mat));
	}

	int GeomLib::DerivMatrixVersion( double t, const double y[], double dydt[], void* params)
	{
		MasterParameter* p_par = static_cast<MasterParameter*>(params);
		const AbstractOdeSystem* p_sys = p_par->_p_system;

		NumtoolsLib::QaDirty<double>& mat = *(p_par->_p_mat);


		InitializeMatrix
		(
			mat,
			*p_par->_p_vec_set,
			p_sys->InterpretationBuffer(),
			p_sys->Par()._par_pop._theta,
			p_sys->Par()._par_pop._V_reset,
			p_sys->IndexResetBin()
		);

		Number n_bins = p_par->_p_system->NumberOfBins();

		for (Index row = 0; row < n_bins; row++){
			dydt[p_sys->MapPotentialToProbabilityBin(row)] = 0.0;
			for (Index column = 0; column < n_bins; column++)
				dydt[p_sys->MapPotentialToProbabilityBin(row)] += mat(row,column)*y[p_sys->MapPotentialToProbabilityBin(column)];
		}

		assert(SumColumnsZero(*p_par->_p_mat));
		assert(ProbabilityConservation(dydt ,n_bins));

		return GSL_SUCCESS;
	}
