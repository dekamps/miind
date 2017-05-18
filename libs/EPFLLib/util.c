#include <stdlib.h>
#define FREE_ARG char*

double *dvector(long n)
/* allocate a double vector with subscript range v[0..n]
   initialize with zeros
 */
{
  double *v;
  v= (double *)malloc((size_t) (n*sizeof(double)));
  /* long i; */
  /* for (i=0;i<n;i++) v[i]=0; */
  return v;
}


int *ivector(long n)
/* allocate a double vector with subscript range v[0..n] */
{
  int *v;
  v=(int *)malloc((size_t) (n*sizeof(int)));
  /* long i; */
  /* for (i=0;i<n;i++) v[i]=0; */
  return v;
}

unsigned int *uivector(long n)
/* allocate a double vector with subscript range v[0..n] */
{
  unsigned int *v;
  v=(unsigned int *)malloc((size_t) (n*sizeof(unsigned int)));
  /* long i; */
  /* for (i=0;i<n;i++) v[i]=0; */
  return v;
}

double **dmatrix(long nrow, long ncol)
/* allocate an int matrix with subscript range m[0..nr][0..nc] */
{
  long i;
  double **m;
  
  /* allocate pointers to rows */
  m=(double **) malloc((size_t)((nrow)*sizeof(double*)));
  
  /* allocate rows and set pointers to them */
  m[0]=(double *) malloc((size_t)((nrow*ncol)*sizeof(double)));
  for(i=1;i<nrow;i++) m[i]=m[i-1]+ncol;
  
  /* for (i=0;i<nrow;i++) */
  /*   for (j=0;j<ncol;j++) */
  /*     m[i][j]=0; */

  /* return pointer to array of pointers to rows */
  return m;
}

int **imatrix(long nrow, long ncol)
/* allocate an int matrix with subscript range m[0..nr][0..nc] */
{
  long i;
  int **m;
  
  /* allocate pointers to rows */
  m=(int **) malloc((size_t)((nrow)*sizeof(int*)));
  
  /* allocate rows and set pointers to them */
  m[0]=(int *) malloc((size_t)((nrow*ncol)*sizeof(int)));
  for(i=1;i<nrow;i++) m[i]=m[i-1]+ncol;
  
  /* return pointer to array of pointers to rows */
  return m;
}










void free_ivector(int *v)
/* free an int vector allocated with ivector() */
{
	free((FREE_ARG) (v));
}

void free_uivector(unsigned int *v)
/* free an int vector allocated with uivector() */
{
	free((FREE_ARG) (v));
}


void free_dvector(double *v)
/* free a double vector allocated with dvector() */
{
	free((FREE_ARG) (v));
}



void free_dmatrix(double **m)
/* free a double matrix allocated by dmatrix() */
{
	free((FREE_ARG) (m[0]));
	free((FREE_ARG) (m));
}

void free_imatrix(int **m)
/* free an int matrix allocated by imatrix() */
{
	free((FREE_ARG) (m[0]));
	free((FREE_ARG) (m));
}
