#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define  Max(a,b) ((a)>(b)?(a):(b))

//#define  N   (2*2*2*2*2*2+2)       // 66
//#define  N   (2*2*2*2*2*2*2+2)     // 130
#define  N   (2*2*2*2*2*2*2*2+2)   // 258
//#define  N   (2*2*2*2*2*2*2*2*2+2) // 514

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

	// 
	double A [N][N][N],  B [N][N][N];
	
	init(A);

	// starting timer
	clock_t timer_start = clock();

	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax(A, B);
		resid(A, B);
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	// stopping timer after relax() + resid()
	clock_t timer_end = clock();
	double time_spent = (double)(timer_end - timer_start) / CLOCKS_PER_SEC;

	verify(A);

	// printing the calculated time
	printf("relax() + resid() were running for {%f} seconds with N = {%d}\n", time_spent, N);

	return 0;
}

void init(double A [N][N][N])
{
	int i,j,k;

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
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s=s+A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);
}
