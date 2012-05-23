#ifndef SCHEDULER
#define SCHEDULER
#include <vector>
#include <iostream>
#include "RandSimulator.h"


class Scheduler
{
	private:

		int _last_traded;
		int _last_f_traded;
		int _last_c_traded;



		int InterleavedSeq(){
			// first time ever
			if(_last_traded < 0)
			{
				_last_traded = f+0;
				_last_f_traded = 0;
				return _last_traded;
			}

			// last was chartist
			if(_last_traded >= c)
			{
				if(_last_f_traded == _n_funds-1)
				{
					_last_f_traded = 0;
					_last_traded = f + 0;
					return _last_traded;
				}
				else
				{
					_last_traded = f + (++_last_f_traded);
					return _last_traded;
				}
			}
			// last was fundamentalist
			else
			{
				if(_last_c_traded == _n_charts-1)
				{
					_last_c_traded = 0;
					_last_traded = c + 0;
					return _last_traded;
				}
				else
				{
					_last_traded = c + (++_last_c_traded);			
					return _last_traded;
				}
			}
		};


		int InterleavedRand(){

			if(_last_traded < 0)
			{
				_last_traded = Random();
				return _last_traded;
			}

			int n = AgentLib::RandSimulator()%8;
			if(_last_traded>=c)
				{
					_last_traded=f+n;
					return _last_traded;
				}
			else
				{
					_last_traded=c+n;
					return _last_traded;
				}
		};

		int Random(){
			int n = AgentLib::RandSimulator()%16;
			if(n>=8)
				return c+n-8;
			if(n<8)
				return f+n;
			exit(1);
		};

	public:

		static const int f = 1001;
		static const int c = 2001;

		// sched_mode: 0-sequential interleaved, 1-random interleaved, 2-random
		Scheduler(int n_funds, int n_charts, int sched_mode)
			{
				_n_funds = n_funds;
				_n_charts = n_charts;
				_sched_mode = sched_mode;
				_last_traded = -1;
				_last_c_traded = -1;
				_last_f_traded = -1;
			};	
		Scheduler( void ){};
		~Scheduler( void ){};

		int GetTraderId()
		{
			switch(_sched_mode){
				case 0: return InterleavedSeq();
				case 1: return InterleavedRand();
				case 2: return Random();
				default:
					exit(1);
			}	
		}

	
		

	protected:
		
		int _n_funds;
		int _n_charts;
		int _sched_mode;

};
#endif