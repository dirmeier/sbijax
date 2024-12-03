
install.packages("devtools")
devtools::install_github("massimilianotamborrino/sdbmpABC")

library(sdbmpABC)


?sdbmsABC::Splitting_JRNMM_output_Cpp()
cut<-10^3 #number of kept samples (relates to the threshold epsilon)
N<-2.5*10^6 #number of iterations
M<-30  #number of observed datasets (corresponds to paths of the output process)
w<-1930.17 #weight for the distance calculation
T<-8 #time interval for the datasets
h<-T/1024
h
#----------------------------------------------------------
numDist<-1
grid<-seq(from=0, to=T, by=h) #time grid

#----------------------------------------------------------

#theta_true: parameters used to simulate the reference data
sig<-2000.0
mu<-220.0
C<-135.0

#fixed model coefficients
A<-3.25
B<-22.0
a<-100.0
b<-50.0
v0<-6.0
vmax<-5.0
r<-0.56

#starting value X(0)
#The process X converges exponentially fast to its invariant regime.
#Thus, the choice of X(0) is not crucial.
#Its impact on the distribution of X diminishes exponentially fast.
startv<-c(0.08,18,15,-0.5,0,0)


set.seed(2)
Splitting_JRNMM_output_Cpp(
    h,
    startv,
    grid,
    exp_matJR(h,a,b),
    t(chol(cov_matJR(h,c(0,0,0,0.01,sig,1),a,b))),
    mu,
    C,
    A, 
    B, 
    a, 
    b, 
    v0, 
    r, 
    vmax
)

