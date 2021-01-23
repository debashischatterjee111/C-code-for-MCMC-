#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h> 
#include <unistd.h>
# include <complex.h>
# include <string.h>

//# include "random.h"



#define PI         3.14159265

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
     
     
     #define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
     
     
     

double r_norm(double mean, double std_dev);  // Returns a normal rv
double rand_val(int seed);                 // Jain's RNG


double  mean_Y(int k, int m);
double sd_Y(int k, int m);
double  sum_X( int n) ;
double  stpt[2]={1.2,1.2};


FILE     *fppX_out;               // File pointer to output file
  char     instring[80];   

   // double get_random() { return ((double)rand() / (double)RAND_MAX); }
int **mat;
int **Y;
double **res,**res10, **res100, **res1000;
double c = 20;
 int k=1; 
double y_bar,y_std,x_k, a;

int MCMC_SIZE;
double aa, mm;
// RAND_MAX=99999;
    double aaa, mmm;
    int N, M ,n, m;
	double *X, *th0, T0; 
	//double T1[2], T2[2];
	double T[2],g[2];
	
	
	double r_norm(double mean=0, double std_dev=1)
{
  double   u, r, theta;           // Variables for Box-Muller method
  double   x;                     // Normal(0, 1) rv
  double   norm_rv;               // The adjusted normal rv

  // Generate u
  u = 0.0;
  while (u == 0.0)
    u = rand_val(0);

  // Compute r
  r = sqrt(-2.0 * log(u));

  // Generate theta
  theta = 0.0;
  while (theta == 0.0)
    theta = 2.0 * PI * rand_val(0);

  // Generate x value
  x = r * cos(theta);

  // Adjust x value for specified mean and variance
  norm_rv = (x * std_dev) + mean;

  // Return the normally distributed RV value
  return(norm_rv);
}

	
	
	double rand_val(int seed)
{
  const long  a =      16807;  // Multiplier
  const long  m = 2147483647;  // Modulus
  const long  q =     127773;  // m div a
  const long  r =       2836;  // m mod a
  static long x;               // Random int value
  long        x_div_q;         // x divided by q
  long        x_mod_q;         // x modulo q
  long        x_new;           // New x value

  // Set the seed if argument is non-zero and then return zero
  if (seed > 0)
  {
    x = seed;
    return(0.0);
  }

  // RNG using integer arithmetic
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if (x_new > 0)
    x = x_new;
  else
    x = x_new + m;

  // Return a random value between 0.0 and 1.0
  return((double) x / m);
}


	
double unif()
{//srand(time(NULL));
return (rand_val(0) );
}
   
   
   double H(double d)
{
	return(d/(1+d));
}
   
   int r_bernoulli( double p)
{ double u;
   u = unif() ;
//printf("\n %lf \n", u);
  if (u < p)
    {
      return 1 ;
    }
  else
    {
      return 0 ;
    }
}

   

double  r_uniform_X()
{
	int b;
double r;
 
srand(time(NULL));
 
//printf("%d \n",RAND_MAX);
 
for (b=0;b<N;b++)
{
 
        r=(rand() / (RAND_MAX + 1.0) * (aaa - aa) + aa);
       //r=(r/ ((RAND_MAX + 1.0) * (aaa - aa) )+ aa);
       X[b]=r;
       // printf("%lf\n", X[b]);
 
        }
	
	
}

double  r_uniform_th0()
{
//	int b;
double rr;
 

 
//printf("%d \n",RAND_MAX);
 
        rr=(unif() * (mmm - mm) + mm);
       //r=(r/ ((RAND_MAX + 1.0) * (aaa - aa) )+ aa);
       th0[0]=rr;
       // printf("%lf\n", X[b]);
 
        
	
}


double  r_Y()
{
	int b,i,j;
double rr;
 
mat = (int **)calloc(N,sizeof(int*));
 for(i=0;i<N;i++)
   {mat[i]=(int *)calloc(M,sizeof(int));
   }
   
   
   Y = (int **)calloc(N,sizeof(int*));
 for(i=0;i<N;i++)
   {Y[i]=(int *)calloc(M,sizeof(int));
   }

 for(i=0;i<N;i++)
   {
   for(j=0;j<M;j++)
     {Y[i][j]= r_bernoulli(H(X[i]));
     mat[i][j]=Y[i][j];
     }
   }   
 
        
	
}



 
void input1()
{ 

    int i, j;
  
  printf("\t >> Please enter the Number of Observations (n): \n");
    scanf("%d", &N);  
    printf("\t \t  Number of Observations (n = %d)  \n", N);
    
       th0 = (double*) malloc(1*sizeof(double));
     printf("\t First, Please enter the limits of Uniform distribution for Theta :\n");
     printf("\t \t The lower limit: ");
     scanf("%lf", &mm);
      printf("\t \t The Upper limit: ");
      scanf("%lf", &mmm);  
             printf("----->> Hence  we will simulate theta ~ Unif[%lf, %lf]\n",mm,mmm);
    printf("\n \t *** Theta (th0) is :\t");
      r_uniform_th0();
        
 T0=th0[0];
              printf("%lf\n", T0);
 
    
     
    
    X = (double*) malloc(N*sizeof(double));
     printf("\t Next, Please enter the limits of Uniform distribution for X :\n");
     printf("\t \t The lower limit: ");
     scanf("%lf", &aa);
      printf("\t \t The Upper limit: ");
      scanf("%lf", &aaa);  
             printf("----->> Hence  we will simulate X ~ Unif[%lf, %lf]\n",aa,aaa);
    printf("\n \t *** First 10 elements of Covariate X is :\n");
      r_uniform_X();
      
    for (i=0;i<10;i++)
{
 
              printf("%lf\n", X[i]);
 
        }
     
    
    for(i=0;i<N;i++)
   {printf("%lf\t", X[i]);
   if(i=N-1)
   {printf("\n-----------------\n");
   }
   }   
    
    sleep(1);
     printf("\t >> Enter the Number of Repeat for any covariate (m): \n");
    scanf("%d", &M);  
    printf("\t \t  Number of Repeat for any covariate (m): (m = %d) : \n", M);
 
 
 
 
 
 printf("\n >>> **  Hence The Order of response matrix Y will be (%d * %d) \n", N, M);
 
}
/* mat = (int **)calloc(N,sizeof(int*));
 for(i=0;i<n;i++)
   {mat[i]=(int *)calloc(M,sizeof(int));
   }

 for(i=0;i<N;i++)
   {for(j=0;j<M;j++)
     {scanf("%d",&mat[i][j]);
     }
   }   
 
 
 */



void input2()
{ int i,j;
 printf("\n\t** Simulating Y[i][j] **\n"); 

 r_Y();
 

printf("\n\t**** Displaying first (10*10) elements of Y ****\n"); 
for (i = 0; i < 10; i++)
    {
        for (j = 0; j < 10; j++)
        {
            printf("%d\t", mat[i][j]);
        }
        printf("\n");
    }
    
    y_bar = mean_Y(1,10);
    //printf("%lf", y_bar);
    
	y_std=sd_Y(1,10);
    //printf("%lf", y_std);
}


double  mean_Y(int k, int m) {
  double sum;
   int  loop;
  double avg;

   sum = avg = 0;
   
   for(loop = 0; loop < m; loop++) {
      sum = sum + Y[k][loop];
   }
   
   avg = (double)sum / loop;
   //printf("Average of array values is %.2f", avg);   
   
   return (avg);
}



double  sum_X( int n) {
  double sum;
   int  loop;
 // double avg;

   sum = 0;
   
   for(loop = 0; loop < n; loop++) {
      sum = sum + X[loop];
   }
   
  
   //printf("Average of array values is %.2f", avg);   
   
   return (sum);
}




double sd_Y(int k, int m)
{
  
  int  i, Number;
  double Mean, Variance, SD, Sum=0, Differ, Varsum=0;
Number=m;
/* printf("\nPlease Enter the N Value\n");
  scanf("%d", &Number);

  printf("\nPlease Enter %d real numbers\n",Number);
 for(i=0; i<Number; i++)
{
     scanf("%f", &Price[i]);
   }
*/
  for(i=0; i<Number; i++)
   {
     Sum = Sum + Y[k][i];
   }
  
  Mean = Sum /(double)Number;

  for(i=0; i<Number; i++)
   {
     Differ = Y[k][i] - Mean;
     Varsum = Varsum + pow(Differ,2);
   }
  
  Variance = Varsum / (double)Number;
  SD = sqrt(Variance);
  
  //printf("Mean               = %.2f\n", Mean);
  //printf("Varience           = %.2f\n", Variance);
 return(SD);
}




double fratio(){
 double th1,vecx1,th2,vecx2;
 double f11, f22, sum2;
 int h;
  th1=g[0];
  vecx1=g[1];
   // printf("%lf+++++\n", vecx1);
  th2=T[0];
  vecx2=T[1];
   // printf("%lf\n", vecx2);
 f11=((m*mean_Y(1,M)*log((th1*vecx1)/(th2*vecx2)))-(M*log((1+th1*vecx1)/(1+th2*vecx2))));
 // printf("%lf\n", f11);
//  print(f11)
     sum2= 0;
     for(h=1;h<N;h++)
     {sum2=sum2+(m*mean_Y(h,M)*log(th1/th2))-(m*log((1+th1*X[h])/(1+th2*X[h])));
      // printf("%lf_____\n", sum2);
     }
     f22= sum2;
 // printf("%lf ###_-_-_-_-\n", f22);
   return(exp(f11+f22));
}


void qqq () 
{double e, th,x_st,u,v,new_x_st,new_th;
 //double g[2];
  e = fabs(r_norm(0,1));
 // printf("\n ...Testing for rnorm: %lf\n", e);
  th=T[0];
  x_st=T[1];
  //#rnorm(1, x, 0.1)
  u = unif();
  if( u<0.5 )
  { new_x_st = x_st+(0.08*e);
  }
  else
  { new_x_st = max(0.001, x_st-(0.08*e));
  }
  
  //#new_eta[1] = rnorm(1,0,1)+new_b*new_x_star
  
   v = unif();
  if( v<0.5 )
  { new_th = th +0.08*e ; 
  }
  else
  { new_th = max(0.001, th-0.08*e ); 
  }
   g[0]=new_th;
   g[1]= new_x_st;
 // printf("%lf++\n",g[1]);
  
  }



void step() 
 {
 
  double lower, upper, alpha,rrr,rr;
  //Pick new point
  qqq();
  
  //h=tp[1]*tp[2]
  lower = max(0.001,y_bar-c*(y_std/sqrt(m)));
  upper = (y_bar+c*(y_std/sqrt(m)));
  //printf("%lf~~~~\n", lower);
  if( H(g[0]*g[1]) <lower | H(g[0]*g[1])>upper)
  {   g[0] =T[0];
   g[1] =T[1];
  }
  //fratio(tp,T)
  rrr=fratio();
 // printf("%lf", rrr);
  // Acceptance probability:
  alpha = min(1.0,rrr );
// Accept new point with probability alpha:
//printf("\n\t FR**atio: %lf\n", alpha);
rr=unif();
  if (rr < alpha)
    { T[0] =g[0];
    //printf("\n$$ %lf\n", T[0]);
   T[1] =g[1];
	}
//	printf("\n\t\t ^^%lf\n",t[1]);
  // Returning the point:
  //printf("4444444444");
}





void mcmc1()
{int i;
//printf("\n\t-----------------------------------------------\n"); 
 printf("\n\t***** Running of MCMC *****\n"); 
 //printf("\n\t-----------------------------------------------\n\n"); 
     printf("\t >> Enter MCMC_SIZE : ");
    scanf("%d", &MCMC_SIZE);  
    printf("\t \t ... Now Preparing MCMC Run for Size %d ... \n", MCMC_SIZE );
 
 
 
res = (double **)calloc(MCMC_SIZE,sizeof(double*));
 for(i=0;i<MCMC_SIZE;i++)
   {res[i]=(double *)calloc(2,sizeof(double));
   }


//y_bar = mean_Y(k,M);
//y_std = sd_Y(k,M);
//x_k = sum_X(N)-X[k];
 //printf("\n%lf %lf %lf\n", y_bar,y_std ,x_k );
//a = sum(y)+1;

}




void run()
 {int i; 
 double temp[2],now[2],per;
 int nsteps;
	nsteps=MCMC_SIZE;
  printf("\n \t \t .. Starting MCMC run of %d Steps .....  \n", nsteps);
  //printf("\n\t %lf\n",t[1]);
   temp[0]=stpt[0];
   temp[1]=stpt[1];
   T[0]=stpt[0];
   T[1]=stpt[1];
  for (i=0; i< nsteps; i++)
  {     
	  step();
	  now[0] =T[0];
	  now[1] =T[1];
	// t=now;
    // printf("\n\t %lf++\n",now[1]);
    res[i][0] = now[0] ;
      res[i][1] = now[1] ;
      temp[0]=now[0];
   temp[1]=now[1];
       printf("\r\t\t In progress %d th steps...", i);
           fflush(stdout);
           per=(((float)i/nsteps)*100);
    if(i==10|i==100|i==1000|i==2000|i==5000|i==nsteps-1)
    { printf("\n\t\t %lf percent  ..\n",per);
           	}
    if (i == nsteps-1) 
    {
    	printf("\n\t\t ******* Done (100  percent)******!\n");
	}
  }
 
  
}


//ress_10 <- run(c(1.2,1.2),  q, 10000)





int main()
{  FILE     *fp_out, *fp_out_1;               // File pointer to output file
  char     instring[80];          // Input string
  double   num_samples;           // Number of samples to generate
  double   mean, std_dev;         // Mean and standard deviation
  double   norm_rv;               // The adjusted normal rv
  int      i,g;                     // Loop counter


 double test1, test2;
double  r1[2]={2.0,5.0};
double  r2[2]={2.0,5.0};


   // double test1;
     printf("\n\t-----------------------------------------------\n"); 
 printf("\n\t***** LOO-CV Posterior of Covariate X_1 *****\n"); 
 printf("\n\t-----------------------------------------------\n\n"); 
 //sleep(1);
  printf("\n\t Enter the Random number seed ====> ");
  scanf("%s", instring);
  rand_val((int) atoi(instring));
 
// printf("\t ## [Now we can safely return to the main purpose !!! )]  \n ");

// printf("testing for uniform: %lf", test1);
   input1();
   n=N;
   m=M;
 
   input2();
    // ptr = (int*) malloc(N * sizeof(int));
    // test1=fratio(r1,r2);
// test2=unif();
// printf("\t ##  [(This is just for verification )]  \n ");
// printf("\t \t test sample 1 (~ Normal(0,1)) = %lf \n ", test1);
 //printf("\t \t test sample 2 (~ Unif(0,1)) = %lf \n ", test2);
      // call the function to get a different value of n every time
mcmc1();
//test1= fratio(r1, r2);
//printf("\n Testing:%lf %lf\n", r1[1],test1);
   // printf("%f\n", n); 
    sleep(2);
//	printf(" RAND_MAX is : %d\n", RAND_MAX);
	 // print your number
	 
	 run();
	 
	 
	   printf("-------------------------------------------------------- \n");
  printf("-  Writing the Generated samples to file                          - \n");
  printf("-------------------------------------------------------- \n");
  num_samples=MCMC_SIZE-1;
   printf("\t Output file name ==========> ");
  scanf("%s", instring);
  fppX_out = fopen(instring, "w");
  if (fppX_out == NULL)
  {
    printf("ERROR in creating output file (%s) \n", instring);
    exit(1);
  }

  for (i=0; i<num_samples; i++)
  {
    // Generate a normally distributed rv
        // Output the ress value
    fprintf(fppX_out, "%f \n", res[i][1]);
  }

  // Output message and close the distribution and output files
  printf("-------------------------------------------------------- \n");
  printf("-  Done! \n");
  printf("-------------------------------------------------------- \n");
  fclose(fppX_out);
  
    return 0;
}

/* 

double uniform(double a, double b)
{
return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}

*/
