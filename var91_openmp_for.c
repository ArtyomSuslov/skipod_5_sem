#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define  Max(a,b) ((a)>(b)?(a):(b))

//#define  N   (2*2*2*2*2*2+2)       // 66
//#define  N   (2*2*2*2*2*2*2+2)     // 130
//#define  N   (2*2*2*2*2*2*2*2+2)   // 258
#define  N   (2*2*2*2*2*2*2*2*2+2) // 514

double   maxeps = 0.1e-7;
int itmax = 100;
double eps;

void relax();
void resid();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;

	int threads[18] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160};
	double threads_time[18];
	int threads_iter;
    double timer_start;
    double timer_end;

	double A[N][N][N], B[N][N][N];
	
	for (threads_iter=0; threads_iter<18; threads_iter++) 
	{
		double timer_start = omp_get_wtime();

		init(A);

		for(it=1; it<=itmax; it++)
		{
			eps = 0.;
			relax(A, B);
			resid(A, B);
			// printf( "it=%4i   eps=%f\n", it,eps);
			if (eps < maxeps) break;
		}

		verify(A);

		double timer_end = omp_get_wtime();
		double time_spent = timer_end - timer_start;

		threads_time[threads_iter] = time_spent;
	}

	for (threads_iter=0; threads_iter<18; threads_iter++) 
	{
		printf("For N = [%d] and [%d] threads: [%f] seconds\n", 
		       N, threads[threads_iter], threads_time[threads_iter]);

	return 0;
}

void init(double A [N][N][N])
{
	int i,j,k;

	#pragma omp parallel for default(none) shared(A) private(i, j, k)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1) A[i][j][k]= 0.;
		else A[i][j][k]= ( 4. + i + j + k) ;
	}
} 

void relax(double A [N][N][N], double B [N][N][N])
{
	int i,j,k;

	#pragma omp parallel for default(none) shared(A, B) private(i, j, k)
	for(i=2; i<=N-3; i++)
	for(j=2; j<=N-3; j++)
	for(k=2; k<=N-3; k++)
	{
		B[i][j][k]=(A[i-1][j][k]+A[i+1][j][k]+A[i][j-1][k]+A[i][j+1][k]+A[i][j][k-1]+A[i][j][k+1]+
			A[i-2][j][k]+A[i+2][j][k]+A[i][j-2][k]+A[i][j+2][k]+A[i][j][k-2]+A[i][j][k+2])/12.;
	}
}

void resid(double A [N][N][N], double B [N][N][N])
{
	int i,j,k;

	#pragma omp parallel for default(none) shared(A, B) private(i, j, k) reduction(max: eps)
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		double e;
		e = fabs(A[i][j][k] - B[i][j][k]);         
		A[i][j][k] = B[i][j][k]; 
		eps = Max(eps,e);
	}
}

void verify(double A [N][N][N])
{
	int i,j,k;

	double s;
	s=0.;
	#pragma omp parallel for default(none) shared(A) private(i, j, k) reduction(+:s) 
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);
}
