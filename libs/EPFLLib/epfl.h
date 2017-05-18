#ifndef _CODE_LIBS_EPFLLIB_INCLUDE_GUARD
#define _CODE_LIBS_EPFLLIB_INCLUDE_GUARD

//changes to version glm_popdyn_1.0_test2:
//lookuptable for G(x)=exp(-0.5*DT*c*exp(x/deltaV)) instead of lambda and Plam


#ifndef GLM
#define GLM 0
#endif


#ifndef GLM2
#define GLM2 2
#endif

#ifndef GLIF
#define GLIF 10
#endif

#ifndef GLIF1
#define GLIF1 11
#endif

#ifndef GLIF2
#define GLIF2 12
#endif

#ifndef GLIF4
#define GLIF4 14
#endif


#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>


#define maxisi 2.
#define PFIRE_MAX 0.99999
#define PFIRE_MIN 0.00001


struct PopParametersGLM{
  double tref;
  double taum;
  double *taus1; //incoming synaptic time constants, decay time of 1st filter
  double *taus2; //incoming synaptic time constants, decay time of 2nd filter
  double *taur1; //incoming synaptic time constants, rise time of 1st filter
  double *taur2; //incoming synaptic time constants, rise time of 2nd filter
  double *a1;    //relative weight of fast synaptic current 
  double *a2;    //relative weight of slow synaptic current 
  double mu;
  double c;
  double deltaV;
  double delay;
  double vth;
  double u_reset;      //reset potential for u in glif mode
  int N;         //number of neurons in population
  double *J;     //incoming synaptic weights
  double *Iext;  //external input (in mV), i.e. Iext is actually R*I
  double *tau_theta;
  double *J_theta; //amplitudes of the N_theta exponential kernels in mV
  int N_theta;  //number of exponentials for dynamic threshold

  //internal constants
  int K;
  int end;
  int indx;
  double Em;
  double Em2;
  double *E_theta;  //Jtheta*exp(-t/tau_theta)};
  double *E_theta2;  //exp(-dt/tau_theta)};
  double *gamma_QR;  //deltaV*(1-exp(-gamma/deltaV))/N; gamma=sum_k Jtheta_k*exp(-t/tautheta_k)
  double *w1;          // effective weight
  double *w2;
  double g;
  double g2;
  double dV;  // resolution of voltage grid for lookup table of hazard/firing prob
  double Vmin;       //min value of voltage grid
  double Vmax;       //max value of voltage grid (this value is not included, i.e. max is Vmax-dv)
  double *Gtable;      //lookup table for firing prob P=1-exp(-lam*DT)
};


struct PopVariablesGLM{
  double h;
  double *u;
  double *Xsyn1; //incoming synaptic currents (filtered with fast decay time, e.g. AMPA, GABA)
  double *Xsyn2; //incoming synaptic currents (filtered with slow decay time, e.g. NMDA)
  double *Isyn1; //total incoming synaptic current (sum of Xsyn1+Xsyn2 filtered with short rise-time)
  double *Isyn2; //total incoming synaptic current (sum of Xsyn1+Xsyn2 filtered with short rise-time)
  unsigned int *n;
  double *m0;
  double *theta;  //threshold kernel
  double *g; //QR threshold variables
  double *v;
  double z;
  double x;
  double Gfree;
  double *G;
};



////////////////////////////////////////////////////////////////////////////////
double **dmatrix(long nrow, long ncol);
double *dvector(long n);
void free_dvector(double *v);
void free_dmatrix(double **m);
void free_ivector(int *v);
unsigned int *uivector(long n);
void free_uivector(unsigned int *v);


double get_mean(double *x,int n);
void print_matrix1(int N,int M, double A[][M]);
void print_matrix2(double **A,int N,int M);
void print_matrix3(int N,int M, double *A[N]);
void print_pop_parameters(struct PopParametersGLM p[], int Npop);
unsigned long int random_seed();
void init_synaptic_filters(double **Es1, double **Es2, double **Er1, double **Er2, struct PopParametersGLM p[], int Npop);
void get_inputs(double *input, int k,double **Es1,double **Es2,double **Er1,double **Er2, struct PopParametersGLM p[], struct PopVariablesGLM pop[],int Npop);
double cond_rate(double x,struct PopParametersGLM *p);
void update(struct PopVariablesGLM *pop, double *nmean, double input,gsl_rng *rng,struct PopParametersGLM *p, int mode);
void simulate(int Nbin,double **A,double **rate,struct PopVariablesGLM pop[],gsl_rng *rng,struct PopParametersGLM p[],int Npop, int mode, int dispprog);
double threshold_kernel(double t, struct PopParametersGLM *p);
void init_glm(struct PopParametersGLM p[],struct PopVariablesGLM pop[],int Npop, gsl_rng *rng, int mode);
void free_pop(struct PopVariablesGLM pop[],struct PopParametersGLM p[],int Npop, int mode);
void get_history_size(struct PopParametersGLM p[],int Npop);
void get_psd_pop(double **SA,int Nbin,int Ntrials,struct PopParametersGLM neuron[],int Npop, int mode);
void get_trajectory(double **A,double **rate,int Nbin, int Npop,struct PopParametersGLM neuron[],double dt, double dtbin, int mode, int seed);
void init_population(struct PopParametersGLM p[], int Npop,double tref[], double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[]);
void clean_population(struct PopParametersGLM p[], int Npop);
void get_psd_with_fullparameterlist(double **SA, int Nbin, int Ntrials, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double Jref[], double *J_theta[], double *tau_theta[], double sigma[], double dt,double dtbin, int mode);
void get_psd_with_2D_arrays(int Nf, double SA[][Nf], int Ntrials, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double Jref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode);
void get_trajectory_with_fullparameterlist(double **A, double **rate, int Nbin, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double Jref[], double *J_theta[], double *tau_theta[], double sigma[], double dt,double dtbin, int mode, int seed);
void get_trajectory_with_2D_arrays(int Nbin, double A[][Nbin], double rate[][Nbin], int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double s[][Nbin], int N_theta[], double Jref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode, int seed);
////////////////////////////////////////////////////////////////////////////////








# endif // include guard
