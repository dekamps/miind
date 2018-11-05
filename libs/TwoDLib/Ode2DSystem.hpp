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
//      If you use this software in work leading to a scientific publication, you should include a reference there to
//      the 'currently valid reference', which can be found at http://miind.sourceforge.net

#ifndef _CODE_2DLIB_ODE2DSYSTEM_INCLUDE_GUARD
#define _CODE_2DLIB_ODE2DSYSTEM_INCLUDE_GUARD

#include <iostream>
#include <numeric>
#include <vector>
#include <math.h>
#include "MPILib/include/TypeDefinitions.hpp"
#include "MPILib/include/ProbabilityQueue.hpp"
#include "Mesh.hpp"
#include "modulo.hpp"
#include "Redistribution.hpp"

namespace TwoDLib {

	class TransitionMatrix;


	  /**
	   * \brief Responsible for representing the density and deterministic evolution.
	   *
	   * An Ode2DSystem is created with a Mesh, a mapping that ensures that density that
	   * runs off a strip is moved to the reversal bin, and a mapping that moves superthreshold density
	   * to a reset bin. The class is also responsible for deterministic evolution:
	   * it maintains the relationship between the density array and the Mesh, and upon
	   */

	class Ode2DSystem {
	public:

		//! Standard Constructor
		Ode2DSystem
		(
			const Mesh&, 					        //!< Mesh in the Python convention. Most models require a reversal bin that is not part of the grid. In that case it must be inserted into the Mesh by calling Mesh::InsertStationary. It is legal not to define an extra reversal bin, and use one of the existing Mesh cells at such, but in that case Cell (0,0) will not exist.
			const std::vector<Redistribution>&,     //!< A mapping from strip end to reversal bin
			const std::vector<Redistribution>&,     //!< A mapping from threshold to reset bin
			const std::vector<Redistribution>&,			//!< A mapping from incoming threshold to reset bin
			const MPILib::Time& tau_refactive = 0   //!< absolute refractive period
		);


		//! Place all initial density in a cell with coordinates (i,j)
		void Initialize(unsigned int, unsigned int);

		//! Map coordinates to a position in the density array. Map(0,0) may be defined or not, this depends on the Mesh and whether Mesh::InsertStationary
		//! was called. The safe way to use Map is in a loop that uses Mesh::NrCellsInStrip and Mesh::NrQuadrilateralStrips.
		unsigned int Map(unsigned int i, unsigned int j) const {return _map[i][j];}

		//! Shift the density
		void Evolve();

		void EvolveWithoutMeshUpdate();
		//! Dump the current density profile (0), or the mass profile (1) to an output stream
		void Dump(std::ostream&, int mode = 0) const;

		//! Redistribute probability that has moved through threshold. Run this after the Master equation
		void RedistributeProbability();

		void RedistributeProbability(MPILib::Number);

		//! Remap probability that has run from the end of a strip. Run this after evolution
		void RemapReversal();

		//! Return the instantaneous firing rate
		double F() const {return _f;}

		//! total probability mass in the system, should not be too far away from 1.0
		double P() const {
			return std::accumulate(_vec_mass.begin(),_vec_mass.end(),0.0)
			+ _reset_refractive.getTotalProbInRefract() + _next_reset.getTotalProbInRefract();
		}

		//! average membrane potential
		double AvgV() const ;

		//! allow direct inspection of the mass array; client must still convert this to a density
		const vector<double>& Mass() const { return _vec_mass; }

		//! allow inspection of the Mesh object
		const Mesh& MeshObject() const { return _mesh; }

		//! absolute refractive period
		MPILib::Time Tau_ref() const { return _tau_refractive; }

		friend class Master;
	    friend class MasterOMP;
	    friend class MasterOdeint;

	    friend void CheckSystem(const Ode2DSystem&, const TransitionMatrix&, const std::vector<Redistribution>&, const std::vector<Redistribution>&, double);

	private:

		//! Copy constructor
		Ode2DSystem(const Ode2DSystem&);

		//! Implement the remapping of probability mass that hits the end of a strip
		class Reversal {
		public:

			Reversal(Ode2DSystem& sys, vector<double>& vec_mass):_sys(sys),_vec_mass(vec_mass){}

			void operator()(const Redistribution& map){
				_vec_mass[_sys.Map(map._to[0],map._to[1])] += _vec_mass[_sys.Map(map._from[0],map._from[1])];
				_vec_mass[_sys.Map(map._from[0],map._from[1])] = 0;
			}

		private:

			Ode2DSystem&	_sys;
			vector<double>& _vec_mass;
		};

		//! Implement the remapping of probability mass that hits threshold
		class Reset {
		public:
			Reset(Ode2DSystem& sys, vector<double>& vec_mass):
				_sys(sys),
				_vec_mass(vec_mass)
			{}

			void operator()(const Redistribution& map){
				double from =  map._alpha*_vec_mass[_sys.Map(map._from[0],map._from[1])];
				_vec_mass[_sys.Map(map._to[0],map._to[1])] += from;
				_sys._f += from;
			}

		private:

			Ode2DSystem&	_sys;
			vector<double>& _vec_mass;
		};

		//! Implement a reset method that observes a refractive period
		class ResetRefractive {
		public:

			ResetRefractive
			(
				Ode2DSystem&                  sys,
				vector<double>&               vec_mass,
				const MPILib::Number&         it,
				MPILib::Time                  tau_refractive,
				const vector<Redistribution>& vec_reset
			):
			_it(it),
			_t_step(sys.MeshObject().TimeStep()),
			_tau_refractive(tau_refractive),
			_vec_reset(vec_reset),
			_vec_queue(vec_reset.size(),sys.MeshObject().TimeStep()),
			_sys(sys),
			_vec_mass(vec_mass)
			{
			}

			void operator()(const Redistribution& map)
			{
				MPILib::Time t = _it*_t_step;
				double from =  map._alpha*_vec_mass[_sys.Map(map._from[0],map._from[1])];
				_sys._f += from;

				MPILib::Index i = &map - &_vec_reset[0];
				MPILib::populist::StampedProbability prob;
				prob._prob = from;
				prob._time = t + _tau_refractive;
				_vec_queue[i].push(prob);

				from = _vec_queue[i].CollectAndRemove(t);
				_vec_mass[_sys.Map(map._to[0],map._to[1])] += from;
			}

			double getTotalProbInRefract() const{
				double total = 0.0;
				for (MPILib::populist::ProbabilityQueue q : _vec_queue){
					total += q.TotalProbability();
				}
				return total;
			}

		private:

			const MPILib::Number&                      _it;
			MPILib::Time                               _t_step;
			MPILib::Time                               _tau_refractive;

			const vector<Redistribution>&              _vec_reset;
			vector<MPILib::populist::ProbabilityQueue> _vec_queue;

			Ode2DSystem&	                           _sys;
			vector<double>&                            _vec_mass;

		};

		class ResetNext {
		public:

			ResetNext
			(
				Ode2DSystem&                  sys,
				vector<double>&               vec_mass,
				const MPILib::Number&         it,
				MPILib::Time                  tau_refractive,
				const vector<Redistribution>& vec_reset
			):
			_it(it),
			_t_step(sys.MeshObject().TimeStep()),
			_tau_refractive(tau_refractive),
			_vec_reset(vec_reset),
			_vec_queue(vec_reset.size(),sys.MeshObject().TimeStep()),
			_sys(sys),
			_vec_mass(vec_mass)
			{
			}

			void operator()(const Redistribution& map, MPILib::Number n_steps)
			{
				MPILib::Time t = _it*_t_step;
				double from =  map._alpha*_vec_mass[_sys.Map(map._from[0],map._from[1])];
				_sys._f += from;

				MPILib::Index i = &map - &_vec_reset[0];
				MPILib::populist::StampedProbability prob;
				prob._prob = from;
				prob._time = t + _tau_refractive + (_t_step * n_steps);

				_vec_queue[i].push(prob);

				from = _vec_queue[i].CollectAndRemove(t);
				_vec_mass[_sys.Map(map._to[0],map._to[1])] += from;
			}

			double getTotalProbInRefract() const{
				double total = 0.0;
				for (MPILib::populist::ProbabilityQueue q : _vec_queue){
					total += q.TotalProbability();
				}
				return total;
			}

		private:

			const MPILib::Number&                      _it;
			MPILib::Time                               _t_step;
			MPILib::Time                               _tau_refractive;

			const vector<Redistribution>&              _vec_reset;
			vector<MPILib::populist::ProbabilityQueue> _vec_queue;

			Ode2DSystem&	                           _sys;
			vector<double>&                            _vec_mass;

		};

		//! Implement cleaning of the probability that was at threshold. TODO: this is mildly inefficient,
		//! in a serial implementation, but seems simpler in threads

		class Clean {
		public:
			Clean(Ode2DSystem& sys, vector<double>& vec_mass):_sys(sys),_vec_mass(vec_mass){
			}

			void operator()(const Redistribution& map){
				_vec_mass[_sys.Map(map._from[0],map._from[1])] = 0.;
			}

		private:
			Ode2DSystem&	_sys;
			vector<double>&	_vec_mass;
		};

		vector<MPILib::Index> InitializeLength(const Mesh&) const;
		vector<MPILib::Index> InitializeCumulative(const Mesh&) const;
		vector<double>        InitializeArea(const Mesh&) const;
		vector<double>        InitializeMass() const;
		bool                  CheckConsistency() const;

		vector< vector<MPILib::Index> > InitializeMap() const;
		void                  UpdateMap();


		const Mesh&           _mesh;
		vector<MPILib::Index> _vec_length;
		vector<MPILib::Index> _vec_cumulative;
	public:
		vector<double>	      _vec_mass;
	private:
		vector<double>		  _vec_area;

		unsigned int	_it;
		double			_f;

public:
		vector<vector<MPILib::Index> > _map;
private:
		vector<Redistribution> _vec_reversal;
		vector<Redistribution> _vec_reset;
		vector<Redistribution> _vec_next_reset;
		Reversal               _reversal;
		Reset                  _reset;
		ResetNext 					 	 _next_reset;
		ResetRefractive        _reset_refractive;
		Clean				   _clean;

		MPILib::Time           _tau_refractive;
	};

}
#endif // include guard
