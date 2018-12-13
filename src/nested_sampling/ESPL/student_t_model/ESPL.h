#ifndef DNest4_ESPL
#define DNest4_ESPL

#include "DNest4.h"
#include <valarray>
#include <ostream>
#include "/home/fran/drive/projects/VBBinaryLensing/VBBinaryLensing/lib/VBBinaryLensingLibrary.h"

class ESPL 
{
	private:
        // Model parameters 
		double DeltaF, Fb, t0, u0, tE, rho;

		// Model prediction
		std::valarray<double> mu;

		// Compute the model line given the current values of the model parameters 
		void calculate_mu();

        //VBBinaryLensing object needed to compute the likelihood
        VBBinaryLensing VBBL;

	public:
		// Constructor
		ESPL();

		// Generate the point from the prior
		void from_prior(DNest4::RNG& rng);

		// Metropolis-Hastings proposals
		double perturb(DNest4::RNG& rng);

		// Likelihood function
		double log_likelihood() const;

		// Print to stream
		void print(std::ostream& out) const;

		// Return string with column information
		std::string description() const;
};

#endif

