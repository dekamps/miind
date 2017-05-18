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
#include <math.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "util.c"

#define maxisi 2.
#define PFIRE_MAX 0.99999
#define PFIRE_MIN 0.00001


double extime;
unsigned int ntrial;

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

double DT;
double DTBIN;


////////////////////////////////////////////////////////////////////////////////

double get_mean(double *x,int n);
void print_matrix1(int N,int M, double A[][M]);
void print_matrix2(double **A,int N,int M);
void print_matrix3(int N,int M, double *A[N]);
void print_pop_parameters(struct PopParametersGLM p[], int Npop);
unsigned long int random_seed();



////////////////////////////////////////////////////////////////////////////////



void init_synaptic_filters(double **Es1, double **Es2, double **Er1, double **Er2, struct PopParametersGLM p[], int Npop)
{
  int i,j;
  for (i=0;i<Npop;i++)
    for (j=0;j<Npop;j++) 
      {
	if (p[i].taur1[j]>0) Er1[i][j]=exp(-DT/p[i].taur1[j]);
	else Er1[i][j]=0; 
	if (p[i].taus1[j]>0) Es1[i][j]=exp(-DT/p[i].taus1[j]);
	else Es1[i][j]=0;

	if (p[i].taur2[j]>0) Er2[i][j]=exp(-DT/p[i].taur2[j]);
	else Er2[i][j]=0; 
	if (p[i].taus2[j]>0) Es2[i][j]=exp(-DT/p[i].taus2[j]);
	else Es2[i][j]=0;
      }
}


void get_inputs(double *input, int k,double **Es1,double **Es2,double **Er1,double **Er2, struct PopParametersGLM p[], struct PopVariablesGLM pop[],int Npop)
{
  int i,j,ndelay;

  double A[Npop];
  for (j=0;j<Npop;j++) 
    {
      ndelay=p[j].delay/DT;
//      A[j]= pop[j].n[(p[j].end-1-ndelay+p[j].K)%p[j].K] / (p[j].N * DT);
      A[j]= pop[j].n[(p[j].end-ndelay+p[j].K)%p[j].K] / (p[j].N * DT);
    }

  for (i=0;i<Npop;i++)
    {

      //base line input
      input[i]=p[i].mu;
      
      //external input, assumed to be slow compared to DT
      if (p[i].Iext != NULL) 
	input[i] += p[i].Iext[k] * (1-p[i].Em);  //add external input if provided

      //synaptic input
      for (j=0;j<Npop;j++) 
	{
	  if (p[i].taur1[j]>0)
	    {
	      printf("Warning: rise time not correctly implemented\n");
	      input[i]+=p[i].w1[j]*pop[i].Isyn1[j];

	      if (p[i].taus1[j] > 0)
		{
		  pop[i].Isyn1[j] = pop[i].Xsyn1[j] + (pop[i].Isyn1[j] - pop[i].Xsyn1[j]) * Er1[i][j];
		  pop[i].Xsyn1[j] = A[j] + (pop[i].Xsyn1[j] - A[j]) * Es1[i][j];
		}
	      else 
		  pop[i].Isyn1[j] = A[j] + (pop[i].Isyn1[j] - A[j]) * Es1[i][j];
	    }
	  else //zero-rise time
	    if (p[i].taus1[j] > 0)
	      {
		input[i] += p[i].w1[j] * (A[j] + (p[i].taus1[j] * (pop[i].Xsyn1[j] - A[j]) * Es1[i][j] - (pop[i].Xsyn1[j] * p[i].taus1[j] - A[j] * p[i].taum) * p[i].Em) / (p[i].taus1[j] - p[i].taum));


		pop[i].Xsyn1[j] = A[j] + (pop[i].Xsyn1[j] - A[j]) * Es1[i][j];
	      }
	    else //delta current
	      {
		input[i]+=p[i].w1[j]*A[j] * (1 - p[i].Em);
	      }		
	

	  //same for 2nd filter
	  if (p[i].taur2[j]>0)
	    {
	      input[i]+=p[i].w2[j]*pop[i].Isyn2[j];

	      if (p[i].taus2[j] > 0)
		{
		  pop[i].Isyn2[j] = pop[i].Xsyn2[j] + (pop[i].Isyn2[j] - pop[i].Xsyn2[j]) * Er2[i][j];
		  pop[i].Xsyn2[j] = A[j] + (pop[i].Xsyn2[j] - A[j]) * Es2[i][j];
		}
	      else 
		  pop[i].Isyn2[j] = A[j] + (pop[i].Isyn2[j] - A[j]) * Es2[i][j];
	    }
	  else //zero-rise time
	    if (p[i].taus2[j] > 0)
	      {
		input[i]+=p[i].w2[j]*pop[i].Xsyn2[j];
		pop[i].Xsyn2[j] = A[j] + (pop[i].Xsyn2[j] - A[j]) * Es2[i][j];
	      }
	    else //delta current
	      input[i]+=p[i].w2[j]*A[j];
	}


    }
}


double cond_rate(double x,struct PopParametersGLM *p)
{
  if (p->deltaV > 0) return p->c * exp(x / p->deltaV);
  else //p->D==0, no noise
    {
      if (x < 0) return 0;
      else return 1./DT;
    }
}

static double get_Plam(double V, struct PopParametersGLM *p, double *G_old)
{
  if (V<p->Vmax)
    {
      if (V>=p->Vmin)
	{
	  //int x=(int)((V - p->Vmin) / p->dV);
	  double G = p->Gtable[(int)((V - p->Vmin) / p->dV)];
	  double Plam = 1 - *G_old * G;
	  //printf("V=%g x=%d lam=%g\n",V,x,lam);
	  *G_old = G;
	  return Plam;
	}
      else
	{
	  *G_old=1;
	  return 0.;
	}
    }
  else
    {
      *G_old = 0; 
      return 1.;
    }
}


void update(struct PopVariablesGLM *pop, double *nmean, double input,gsl_rng *rng,struct PopParametersGLM *p, int mode)
{

  int i,k,r;
  int kref=(int)(p->tref/DT);
  double W=0, X=0, Y=0, Z=0, theta=p->vth;
  int end=p->end, K=p->K;
  double threshold;

  // first, determine non-refractory spiking prob.
  for (r=0;r < p->N_theta; r++)
    {
      pop->g[r] = pop->g[r]*p->E_theta2[r] + (1 - p->E_theta2[r]) * pop->n[end]/(p->N * DT);
      theta += p->E_theta[r] * pop->g[r];
    }

  double Pfree;
  pop->h = input + (pop->h - p->mu) * p->Em;
  Pfree=get_Plam(pop->h-theta,p,&(pop->Gfree));

  for (k=0; k<K; k++) X += pop->m0[k];

  double ztot,Plam,PLAM, mm;
  theta -= pop->n[end] * p->gamma_QR[0];//substracts effect of n[end] above, added again in 1st loop below, but ensures that threshold theta(t+DT|k=end) is correct

  switch(mode)
    {
    case GLM:
      for (k=end;k<=end-kref+K;k++)
	{
	  i=k%K;
	  threshold = pop->theta[k-end] + theta;
	  theta += pop->n[i] * p->gamma_QR[k-end];
	  Plam=get_Plam(pop->h-threshold,p,pop->G+i);

	  Z += pop->v[i];
	  mm = Plam * pop->m0[i];
	  Y += Plam * pop->v[i];
	  W += mm;
	  pop->v[i] = (1-Plam)*(1-Plam)*pop->v[i] + mm;
	  pop->m0[i] -= mm;
	}      
      break;

    case GLIF:
      {
      for (k=end;k<=end-kref+K;k++)
	{
	  i=k%K;
	  threshold = pop->theta[k-end] + theta;
	  theta += pop->n[i] * p->gamma_QR[k-end];
	  pop->u[i] = input + (pop->u[i] - p->mu) * p->Em;//GLIF
	  Plam=get_Plam(pop->u[i]-threshold,p,pop->G+i);

	  Z += pop->v[i];
	  mm = Plam * pop->m0[i];
	  Y += Plam * pop->v[i];
	  W += mm;
	  pop->v[i] = (1-Plam)*(1-Plam)*pop->v[i] + mm;
	  pop->m0[i] -= mm;

	}
      }
      pop->u[end] = p->u_reset;//GLIF
      break;
    }
    
  ztot = Z + pop->z;
  if (ztot>0) PLAM=(Y+Pfree*pop->z)/ztot;
  else PLAM=0;
  //  printf("W=%g Z=%g z=%g Y=%g y=%g\n",W, Z,pop->z,Y,Pfree*pop->z);
  
  *nmean = W + Pfree * pop->x + PLAM * (p->N - X - pop->x);
  pop->n[end] = gsl_ran_binomial(rng, *nmean/p->N, p->N);

  pop->z=(1-Pfree)*(1-Pfree)*pop->z+Pfree*pop->x+pop->v[end];
  pop->x=(1-Pfree)*pop->x+pop->m0[end];

  pop->m0[end] = pop->n[end];
  pop->v[end] = 0.;
  pop->G[end] = 1.;

}








void simulate(int Nbin,double **A,double **rate,struct PopVariablesGLM pop[],gsl_rng *rng,struct PopParametersGLM p[],int Npop, int mode, int dispprog)
{
  double input[Npop],normfact[Npop], nmean[Npop], nmean_bin[Npop];
  int n_bin[Npop]; 
  int L,i;
  int l=0,k=0;
  L=(int)(DTBIN/DT+0.5);
  for (i=0;i<Npop;i++) 
    {
      n_bin[i]=0;
      nmean_bin[i]=0;
      normfact[i]=1./(L*DT*p[i].N);
    }
  
  double **Es1=dmatrix(Npop,Npop);
  double **Es2=dmatrix(Npop,Npop);
  double **Er1=dmatrix(Npop,Npop);
  double **Er2=dmatrix(Npop,Npop);
  init_synaptic_filters(Es1,Es2,Er1,Er2,p,Npop);  

  int dispcount=(int)(Nbin/100);

  //  clock_t start=clock();
  while (k<Nbin)
    {
      //      printf("Nbin=%d, k=%d l=%d\n",Nbin,k,l);
      get_inputs(input,k,Es1,Es2,Er1,Er2,p,pop,Npop);
      
      for (i=0;i<Npop;i++)
	{
	  update(pop+i,nmean+i,input[i],rng,p+i,mode);
      	  n_bin[i]+=pop[i].n[p[i].end];
      	  nmean_bin[i]+=nmean[i];
   	}
      l++;
      //           printf("nmeanbin=%g\n",nmean_bin[0]);
	   //      printf("\n");
      if (l>=L)
      	{
      	  for (i=0;i<Npop;i++)
	    {
	      A[i][k] = n_bin[i] * normfact[i];
	      rate[i][k] = nmean_bin[i] * normfact[i];

	      //if (i==0) printf("k=%d rate=%g\n",k,rate[i][k]);
	      //	      aa[i][k]=a[i];
      	      n_bin[i]=0;
	      nmean_bin[i]=0;
	    }
      	  l=0;
      	  k++;
      	}
      
      for (i=0;i<Npop;i++) p[i].end=(p[i].end + 1) % p[i].K;

      if (dispprog)
      	if ((k+1)%dispcount==0)
      	  {
      	    printf("%d%%  \r",(int)((k+1) * 100. / Nbin));
      	    fflush(stdout);
      	  }

    }

  //  printf("Execution time of mesoscopic sim: %g seconds, K=%d\n",(double)(clock()-start)/CLOCKS_PER_SEC,p[0].K);


  free_dmatrix(Es1);
  free_dmatrix(Es2);
  free_dmatrix(Er1);
  free_dmatrix(Er2);

}


double threshold_kernel(double t, struct PopParametersGLM *p)
{
  double theta=0;
  int r;
  for (r=0; r<p->N_theta;r++) 
    if (t>=0)  theta += p->J_theta[r] / p->tau_theta[r] * exp(-t / p->tau_theta[r]);
  return theta;
}



void init_glm(struct PopParametersGLM p[],struct PopVariablesGLM pop[],int Npop, gsl_rng *rng, int mode)
//memory allocation and initialization of variables, 
{
  int i,j,k,r, K;
  for (i=0;i<Npop;i++)
    {
      K=p[i].K;
      //      printf("i=%d K=%d\n",i,K);
      pop[i].n=uivector(K);
      pop[i].m0=dvector(K);
      pop[i].v=dvector(K);
      if (mode>=GLIF) pop[i].u=dvector(K);
      else pop[i].u=NULL;
      pop[i].Isyn1=dvector(Npop);
      pop[i].Isyn2=dvector(Npop);
      pop[i].Xsyn1=dvector(Npop);
      pop[i].Xsyn2=dvector(Npop);

      p[i].Em= exp(-DT / p[i].taum);
      p[i].Em2= exp(-0.5*DT / p[i].taum);
      p[i].end = 0;

      pop[i].theta=dvector(K);
      p[i].gamma_QR=dvector(K);
      pop[i].G=dvector(K);

      for (k=0;k<K;k++)
	{
	  /* if (k<K-1)  */
	  /*   { */
	  /*     pop[i].m0[k]=(double)(p[i].N)/(K-1); //normalization (the K-1 element is set zero below) */
	  /*     pop[i].n[k]=(int)(pop[i].m0[k]+1); //initializing with spike count corresponding to zero hazard */

	  /* pop[i].n[k]=(int)(100*DT*p[i].N+1); //initializing with spike count corresponding to 100 Hz */
	  /* pop[i].m0[k]=exp(-(K-k-1)*DT*100)*pop[i].n[k]; */
	  /* n+=pop[i].m0[k]; */
	  //if (i==0) printf("m0=%g n=%d\n", pop[i].m0[k], pop[i].n[k]);
	  /*   } */

	  pop[i].m0[k]=0;
	  pop[i].n[k]=0;
	  pop[i].v[k]=0;
	  if (mode>=GLIF) pop[i].u[k]=p[i].vth-20*gsl_rng_uniform(rng); 

	  pop[i].theta[k]=threshold_kernel( (K-k-1) * DT, p+i);//shifted by DT to get filter at theta(s+DT)
	  p[i].gamma_QR[k] = p[i].deltaV * (1 - exp(-pop[i].theta[k] / p[i].deltaV)) / p[i].N;
	  //	  if (i==0) printf("i=%d k=%d gamma_QR=%g\n",i,k,p[i].gamma_QR[k]);

	  pop[i].G[k]=1.;
	}
      pop[i].Gfree=1.;

      pop[i].n[K-1]=p[i].N;
      pop[i].m0[K-1]=p[i].N;
      pop[i].z=0.;
      pop[i].x=0.;
      //pop[i].x=(p[i].N - n) * DT;
      /* printf("i=%d n=%g x/dt=%g\n",i,n,pop[i].x/DT); */

      for (j=0;j<Npop;j++) pop[i].Isyn1[j]=0;
      for (j=0;j<Npop;j++) pop[i].Isyn2[j]=0;
      for (j=0;j<Npop;j++) pop[i].Xsyn1[j]=0;
      for (j=0;j<Npop;j++) pop[i].Xsyn2[j]=0;

      pop[i].h=p[i].mu;

      //pre-calculating constant J_k*exp(-tau_rel/tau_k)
      pop[i].g=dvector(p[i].N_theta);
      for (r=0; r < p[i].N_theta; r++)
	pop[i].g[r] = pop[i].n[0]/(p[i].N * DT)/p[i].tau_theta[r];

      p[i].E_theta=dvector(p[i].N_theta);
      p[i].E_theta2=dvector(p[i].N_theta);
      //      printf("E_theta= ");
      for (r=0; r < p[i].N_theta; r++)
	{
	  p[i].E_theta[r] = p[i].J_theta[r] * exp(- K * DT / p[i].tau_theta[r]);
	  p[i].E_theta2[r] = exp(- DT / p[i].tau_theta[r]);
	  //	  printf("%g tautheta=%g",p[i].E_theta[r],p[i].tau_theta[r]);
	}
      //      printf("\n");

      
            //pre-computing firing probability as lookup table
      p[i].dV=p[i].deltaV/100;
      p[i].Vmin = p[i].deltaV * log(-log(1-PFIRE_MIN)/DT / p[i].c);
      p[i].Vmax = p[i].deltaV * log(-log(1-PFIRE_MAX)/DT / p[i].c);
      int L=(p[i].Vmax - p[i].Vmin)/p[i].dV;
      //      printf("Lookup tbl (popul %d): Use L=%d, Vmin=%g Vmax=%g dV=%g\n",i+1,L,p[i].Vmin,p[i].Vmax,p[i].dV);
      p[i].Gtable=dvector(L);
      for (k=0;k<L;k++)
	{
	  double V=p[i].Vmin + k*p[i].dV;
	  p[i].Gtable[k]= exp(-0.5 * p[i].c * exp(V/p[i].deltaV) * DT);
	}
    }
}



void get_history_size(struct PopParametersGLM p[],int Npop)
{
  int i;
  double tmax=20.;//sec

  for (i=0;i<Npop;i++)
    {
      int k=tmax/DT;
      int kmin=5*p[i].taum/DT;
      while ((threshold_kernel(k * DT, p+i) / p[i].deltaV < 0.1) && (k>kmin)) k--;
      //      printf("K=%d\n",k);
      if (k*DT<=p[i].delay)
	k=(int)(p[i].delay/DT)+1;
      //      printf("K=%d\n",k);
      if (k*DT<=p[i].tref)
	k=(int)(p[i].tref/DT)+1;
      p[i].K=k;
      //      p[i].K=20000;
      printf("Use K=%d bins for history of population %d (T=%g sec)\n",p[i].K,i+1,p[i].K*DT);
    }
}




void free_pop(struct PopVariablesGLM pop[],struct PopParametersGLM p[],int Npop, int mode)
{
  int i;
  for (i=0;i<Npop;i++)
    {
      free_uivector(pop[i].n);
      free_dvector(pop[i].m0);
      free_dvector(pop[i].v);
      if (mode>=GLIF) free_dvector(pop[i].u);
      free_dvector(pop[i].Isyn1);
      free_dvector(pop[i].Isyn2);
      free_dvector(pop[i].Xsyn1);
      free_dvector(pop[i].Xsyn2);

      free_dvector(pop[i].g);
      free_dvector(pop[i].theta);
      free_dvector(p[i].E_theta);
      free_dvector(p[i].E_theta2);
      free_dvector(p[i].gamma_QR);
      free_dvector(pop[i].G);
      free_dvector(p[i].Gtable);
    }
}



void get_psd_pop(double **SA,int Nbin,int Ntrials,struct PopParametersGLM neuron[],int Npop, int mode)
{
  gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2);
  unsigned long int seed;
  seed = random_seed();
  gsl_rng_set(rng,seed);
  //  gsl_rng_set (rng,(long)time(NULL));


  get_history_size(neuron,Npop);
  
  int i,j,n;

  struct PopVariablesGLM pop[Npop];
  init_glm(neuron,pop,Npop, rng, mode);
  double sum=0;
  for (i=0;i<neuron[0].K;i++)
    {
      sum+=pop[0].m0[i];
    }


  double **A, **rate;
  A=dmatrix(Npop,Nbin);
  rate=dmatrix(Npop,Nbin);

  simulate(Nbin,A,rate,pop,rng,neuron,Npop, mode, 0);

  double complex *AF[Npop];
  for (j=0;j<Npop;j++)
    {
      AF[j]=(double complex *)malloc(sizeof(double complex)*Nbin);
      SA[j]=(double *)malloc(sizeof(double)*(Nbin/2));//psd of A
      for (i=0;i<Nbin/2;i++) SA[j][i]=0;
    }

  fftw_plan plan=fftw_plan_dft_r2c_1d(Nbin,A[0],AF[0],FFTW_ESTIMATE);

  for (n=0;n<Ntrials;n++)
    {
      //      if (n%10==9)
	{
           printf("trial %d ",n+1);
           for (j=0;j<Npop;j++) printf("%g ",get_mean(A[j],Nbin));
           printf("\r");
	   fflush(stdout);
	}
	simulate(Nbin,A,rate,pop,rng,neuron,Npop,mode, 0);

      for (j=0;j<Npop;j++)
	{
	  fftw_execute_dft_r2c(plan,A[j],AF[j]);
	  for (i=1;i<Nbin/2+1;i++) SA[j][i-1]+=creal(AF[j][i]*conj(AF[j][i]))*DTBIN/Nbin;
	}
    }
  printf("trial %d ",n);
  for (j=0;j<Npop;j++) printf("%g ",get_mean(A[j],Nbin));
  printf("\n");

  for (j=0;j<Npop;j++)
    for (i=0;i<Nbin/2;i++)  SA[j][i]/=Ntrials;


  gsl_rng_free (rng);
  free_pop(pop,neuron,Npop, mode);

  free_dmatrix(A);
  free_dmatrix(rate);

  for (j=0;j<Npop;j++) free(AF[j]);
  fftw_destroy_plan(plan);   
}






void get_trajectory(double **A,double **rate,int Nbin, int Npop,struct PopParametersGLM neuron[],double dt, double dtbin, int mode, int seed)
{

  gsl_rng *rng=gsl_rng_alloc(gsl_rng_taus2);
  gsl_rng_set (rng,(long)seed);
  /* printf("seed=%d\n",seed); */
  /* printf("rndnr=%g\n",gsl_rng_uniform(rng)); */
  DT=dt;
  DTBIN=dtbin;

  get_history_size(neuron,Npop);


  struct PopVariablesGLM pop[Npop];
  init_glm(neuron,pop,Npop, rng, mode);

  clock_t start=clock();
  simulate(Nbin,A,rate,pop,rng,neuron,Npop, mode, 1);
  double sim_t=(double)(clock()-start)/CLOCKS_PER_SEC;
  printf("Execution time of mesoscopic dynamics: %g seconds, %g s per biosecond\n",sim_t,sim_t/Nbin/DTBIN);


  int i;
  //for (i=0;i<20;i++) printf("rate=%g\n",rate[0][i]);
  for (i=0;i<Npop;i++) printf("mean rate of population %d: %g \n",i+1,get_mean(rate[i],Nbin));
  //  for (i=0;i<Nbin;i++) printf("%g\n",rate[0][i]);

  gsl_rng_free (rng);
  free_pop(pop,neuron,Npop, mode);
}







void init_population(struct PopParametersGLM p[], int Npop,double tref[], double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double J_ref[], double *J_theta[], double *tau_theta[], double sigma[])
{

  int k,l;
  for (k=0;k<Npop;k++)
    {
      p[k].indx=k;
      p[k].tref=tref[k];
      p[k].taum=taum[k];
      p[k].mu=mu[k];
      p[k].c=c[k];
      p[k].deltaV=deltaV[k];
      p[k].delay=delay[k];
      p[k].vth=vth[k];
      p[k].u_reset = vreset[k];
      p[k].N=N[k];
      p[k].g=sqrt(1-exp(-2*DT/p[k].taum)) * sigma[k];
      p[k].g2=sqrt(1-exp(-DT/p[k].taum)) * sigma[k];
      p[k].Iext=signal[k];
      p[k].taus1=taus1[k];
      p[k].taus2=taus2[k];
      p[k].taur1=taur1[k];
      p[k].taur2=taur2[k];
      if (J_ref[k]==0)
	p[k].N_theta=N_theta[k];
      else 
	p[k].N_theta=N_theta[k]+1; //putting exponential refractory kernel into threshold (theta kernel)
      

      p[k].J_theta=dvector(p[k].N_theta);
      p[k].tau_theta=dvector(p[k].N_theta);
      for (l=0;l<N_theta[k];l++)
	{
	  p[k].J_theta[l]=J_theta[k][l];
	  p[k].tau_theta[l]=tau_theta[k][l];
	  //	  printf("k=%d l=%d J=%g tau=%g\n",k,l,J_theta[k][l],tau_theta[k][l]);
	}
      if (J_ref[k]!=0.)
	{
	  p[k].J_theta[N_theta[k]] = J_ref[k]*taum[k];
	  p[k].tau_theta[N_theta[k]] = taum[k];
	}

      //effective synaptic weights
      p[k].w1=dvector(Npop);
      p[k].w2=dvector(Npop);
      for (l=0;l<Npop;l++)
	{
	  p[k].w1[l] = J[k][l] * p_conn[k][l] * N[l] * taum[k] * a1[k][l] / (a1[k][l]+a2[k][l]);
	  p[k].w2[l] = J[k][l] * p_conn[k][l] * N[l] * taum[k] * a2[k][l] / (a1[k][l]+a2[k][l]);
	  //printf("k=%d l=%d a1=%g w1=%g\n",k,l,a1[k][l],p[k].w1[l]);
	}
    }

}




void clean_population(struct PopParametersGLM p[], int Npop)
{
  int k;
  for (k=0;k<Npop;k++) free_dvector(p[k].w1);
  for (k=0;k<Npop;k++) free_dvector(p[k].w2);
  for (k=0;k<Npop;k++) free_dvector(p[k].J_theta);
  for (k=0;k<Npop;k++) free_dvector(p[k].tau_theta);
}



void get_psd_with_fullparameterlist(double **SA, int Nbin, int Ntrials, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double Jref[], double *J_theta[], double *tau_theta[], double sigma[], double dt,double dtbin, int mode)
{
  double **signal=dmatrix(Npop,1);
  int i;
  for (i=0;i<Npop;i++) 
    signal[i]=NULL;

  DT=dt;
  DTBIN=dtbin;

  struct PopParametersGLM p[Npop];
  init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, Jref, J_theta, tau_theta, sigma);


  get_psd_pop(SA, Nbin, Ntrials, p, Npop, mode);
  free_dmatrix(signal);
  clean_population(p,Npop);

}



 void get_psd_with_2D_arrays(int Nf, double SA[][Nf], int Ntrials, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], int N_theta[], double Jref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode)
{
  //convert 2D array to double**
  double *SA_tmp[Npop], *J_theta_ptr[Npop], *tau_theta_ptr[Npop];
  int i,j;
  int Nbin=2*Nf;

  int indx=0;
  for (i=0;i<Npop;i++)
    {
      J_theta_ptr[i]=&(J_theta[indx]);
      tau_theta_ptr[i]=&(tau_theta[indx]);
      indx+=N_theta[i];
      /* if (N_theta[i]>0) */
      /* 	{ */
      /* 	  J_theta_ptr[i]=&(J_theta[indx]); */
      /* 	  tau_theta_ptr[i]=&(tau_theta[indx]); */
      /* 	  indx+=N_theta[i]; */
      /* 	} */
      /* else */
      /* 	{ */
      /* 	  J_theta_ptr[i]=NULL; */
      /* 	  tau_theta_ptr[i]=NULL; */
      /* 	} */
    }


  get_psd_with_fullparameterlist(SA_tmp, Nbin, Ntrials, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, N_theta, Jref, J_theta_ptr, tau_theta_ptr, sigma, dt, dtbin, mode);


  for (j=0;j<Npop;j++) 
    for (i=0;i<Nf;i++)
      SA[j][i]=SA_tmp[j][i];

  for (j=0;j<Npop;j++)
    free(SA_tmp[j]);
    
}





void get_trajectory_with_fullparameterlist(double **A, double **rate, int Nbin, int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double **signal, int N_theta[], double Jref[], double *J_theta[], double *tau_theta[], double sigma[], double dt,double dtbin, int mode, int seed)
{
  DT=dt;
  DTBIN=dtbin;

  struct PopParametersGLM p[Npop];
  init_population(p, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, Jref, J_theta, tau_theta, sigma);
  //  print_pop_parameters(p,Npop);


  get_trajectory(A, rate, Nbin, Npop, p, dt, dtbin, mode, seed);

  clean_population(p,Npop);
}




void get_trajectory_with_2D_arrays(int Nbin, double A[][Nbin], double rate[][Nbin], int Npop, double *tref, double taum[], double taus1[][Npop], double taus2[][Npop], double taur1[][Npop], double taur2[][Npop], double a1[][Npop], double a2[][Npop], double mu[], double c[], double deltaV[], double delay[], double vth[], double vreset[], int N[], double J[][Npop], double p_conn[][Npop], double s[][Nbin], int N_theta[], double Jref[], double J_theta[], double tau_theta[], double sigma[], double dt,double dtbin, int mode, int seed)
{

  //convert 2D array to double**
  double *rate_ptr[Npop],*AA[Npop],*signal[Npop], *J_theta_ptr[Npop], *tau_theta_ptr[Npop];
  int i;

  for (i=0;i<Npop;i++)
    AA[i]=A[i];
  for (i=0;i<Npop;i++)
    rate_ptr[i]=rate[i];
  for (i=0;i<Npop;i++)
    signal[i]=s[i];


  int indx=0;
  for (i=0;i<Npop;i++)
    {
      if (N_theta[i]>0)
  	{
  	  J_theta_ptr[i]=&(J_theta[indx]);
  	  tau_theta_ptr[i]=&(tau_theta[indx]);
  	  indx+=N_theta[i];
  	}
      else
  	{
  	  J_theta_ptr[i]=NULL;
  	  tau_theta_ptr[i]=NULL;
  	}
    }


  get_trajectory_with_fullparameterlist(AA, rate_ptr, Nbin, Npop, tref, taum, taus1, taus2, taur1, taur2, a1, a2, mu, c, deltaV, delay, vth, vreset, N, J, p_conn, signal, N_theta, Jref, J_theta_ptr, tau_theta_ptr, sigma, dt, dtbin, mode, seed);

}








//===================================================================
// SRM Model
//===================================================================















































void print_pop_parameters(struct PopParametersGLM p[], int Npop)
{

  int k,l;
  printf("\n");
  for (k=0;k<Npop;k++)
    {
      printf("POPULATION %d\n",k+1);
      printf("tref=%g\n",p[k].tref);
      printf("taum=%g\n",p[k].taum);
      printf("mu=%g\n",p[k].mu);
      printf("c=%g\n",p[k].c);
      printf("DeltaV=%g\n",p[k].deltaV);
      printf("delay=%g\n",p[k].delay);
      printf("vth=%g\n",p[k].vth);
      printf("vreset=%g\n",p[k].u_reset);
      printf("N=%d\n",p[k].N);
      printf("Ntheta=%d\n",p[k].N_theta);
      for (l=0;l<p[k].N_theta;l++)
	{
	  printf("Jtheta=%g\n",p[k].J_theta[l]);
	  printf("tautheta=%g\n",p[k].tau_theta[l]);
	}  
      for (l=0;l<Npop;l++)
	{
	  printf("pop%d to pop%d:\n",l+1,k+1);
	  printf("   w1=%g\n",p[k].w1[l]);
	  printf("   w2=%g\n",p[k].w2[l]);
	  printf("   taus1=%g taur1=%g\n",p[k].taus1[l],p[k].taur1[l]);
	  printf("   taus2=%g taur2=%g\n",p[k].taus2[l],p[k].taur2[l]);
	}
      printf("\n");
    }
  printf("DT=%g DTBIN=%g\n",DT,DTBIN);
}

double get_mean(double *x,int n)
{
  int i;
  double m=0;
  if (n>200) //discard initial transient
    {
      for (i=200;i<n;i++) m+=x[i];
      return m/(n-200);
    }
  else
    {
      for (i=0;i<n;i++) m+=x[i];
      return m/n;
    }
}


void print_matrix1(int N,int M, double A[][M])
{
  int i,j;
  for (i=0;i<N;i++)
    {
      for (j=0;j<M;j++)
	printf("%g ",A[i][j]);
      printf("\n");
    }
}

void print_matrix2(double **A,int N,int M)
{
  int i,j;
  for (i=0;i<N;i++)
    {
      for (j=0;j<M;j++)
	printf("%g ",A[i][j]);
      printf("\n");
    }
}

void print_matrix3(int N,int M, double *A[N])
{
  int i,j;
  for (i=0;i<N;i++)
    {
      for (j=0;j<M;j++)
	printf("%g ",A[i][j]);
      printf("\n");
    }
}


/* unsigned long int random_seed() */
/* { */

/*  unsigned int seed; */
/*  FILE *devrandom; */

/*  if ((devrandom = fopen("/dev/random","r")) == NULL) { */
/*    fprintf(stderr,"Cannot open /dev/random, setting seed to 0\n"); */
/*    seed = 0; */
/*  } else { */
/*    fread(&seed,sizeof(seed),1,devrandom); */
/*    if(verbose == D_SEED) printf("Got seed %u from /dev/random\n",seed); */
/*    fclose(devrandom); */
/*  } */

/*  return(seed); */

/* } */

#include <sys/time.h>
unsigned long int random_seed()
{

 unsigned int seed;
 struct timeval tv;
 FILE *devrandom;

 if ((devrandom = fopen("/dev/urandom","r")) == NULL) {
   gettimeofday(&tv,0);
   seed = tv.tv_sec + tv.tv_usec;
 } else {
   fread(&seed,sizeof(seed),1,devrandom);
   fclose(devrandom);
 }

 return(seed);

}
