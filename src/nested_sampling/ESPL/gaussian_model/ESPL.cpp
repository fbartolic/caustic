#include "Data.h"
#include "ESPL.h"
#include "DNest4.h"
#include "../../../DNest4/code/Distributions/Uniform.h"

using namespace std;

ESPL::ESPL()
: VBBL() // Initialize member object instance of VBBinaryLensing
{
    // Load pre-calculated table needed for ESPL, it needs to be loaded only once
	VBBL.LoadESPLTable("/home/fran/drive/projects/VBBinaryLensing/VBBinaryLensing/data/ESPL.tbl"); 
} 

void ESPL::calculate_mu()
{
	const auto &t = Data::get_instance().get_t();
    valarray<double> tmp (t.size());

    double A_u0 = VBBL.ESPLMag2(u0, rho);

    for(size_t i=0; i < t.size(); ++i) {
        double u = sqrt(u0*u0 + pow((t[i] - t0)/tE, 2.));
        double A_u = VBBL.ESPLMag2(u, rho);
        tmp[i] = DeltaF*(A_u - 1.)/(A_u0 - 1.) + Fb;
    }

    mu = tmp;
}

void ESPL::from_prior(DNest4::RNG& rng)
{
    // Priors for (DeltaF, Fb), U(0,1)
    DeltaF = rng.rand(); 
    Fb = rng.rand(); 

    const auto &t = Data::get_instance().get_t();

    // Prior for t0, U(0.9*tmin, 1.1*tmax) 
    DNest4::Uniform uniform_t0(0.9*t[0], 1.1*t.max());
    t0 = uniform_t0.generate(rng);

    // Prior for u0, U(0, 1.5)
    u0 = 1.5*rng.rand();

    // Prior for tE, U(0.1, 365)
    DNest4::Uniform uniform_tE(0.1, 365.);
    tE = uniform_tE.generate(rng); 

    // Prior for rho, U(0.1, 365)
    DNest4::Exponential exponential_rho(0.01);
    rho = exponential_rho.generate(rng); 

	// Compute the model lightcurve
	calculate_mu();
}

double ESPL::perturb(DNest4::RNG& rng)
{
	double log_H = 0.;

	// Proposals explore the prior, log_H = log(pi(theta')/pi(theta))
    // Only one parameter is perturbed at each step in the parameter space

	int param = rng.rand_int(6);
    switch(param) {
       case 0:{
        DNest4::Uniform uniform(0., 1.);
        log_H += uniform.perturb(DeltaF, rng); // Perturbs DeltaF in-place
        break;
              }

       case 1:{
        DNest4::Uniform uniform(0., 1.);
        log_H += uniform.perturb(Fb, rng); 
        break;
              }

       case 2:{
        const auto &t = Data::get_instance().get_t();
        DNest4::Uniform uniform(t[0], t.max());
        log_H += uniform.perturb(t0, rng); 
        break;
              }

       case 3:{
        DNest4::Uniform uniform(0., 1.5);
        log_H += uniform.perturb(u0, rng); 
        break;
              }

       case 4:{
        DNest4::Uniform uniform(0.1, 365.);
        log_H += uniform.perturb(tE, rng); 
        break;
              }
       case 5:{
        log_H += (1/0.01)*rho;
        DNest4::Exponential exponential(0.01);
        log_H += exponential.perturb(rho, rng); 

        break;
              }

    }
	
	// Calculate mu again since the parameters changed
	calculate_mu();

	return log_H;
}

double ESPL::log_likelihood() const
{
	// Grab the data
	const auto& F = Data::get_instance().get_F();
	const auto& sigF = Data::get_instance().get_sigF();

    // Conventional gaussian sampling distribution 
    double log_L = 0.0;

    for (size_t i = 0; i < F.size(); i++){
        log_L += -0.5*log(2*M_PI*sigF[i]*sigF[i]) - 0.5*pow((F[i] - mu[i])/sigF[i], 2.);
    }

    return log_L;
}

void ESPL::print(std::ostream& out) const
{
	out<<DeltaF<<' '<<Fb<<' '<<t0<<' '<<u0<<' '<<tE<<' '<<rho;
}

string ESPL::description() const
{
	return string("DeltaF, Fb, t0, u0, tE, rho");
}
