- Implement analytically marginalized PSPL model. 
- Implement parallax, decide on default parametrization.
- Figure out how to proceed with the development of the Data handling class, probably 
should utilize Astropy more.
- Figure out systematic way for treating error-bars.
- Decide on sensible default kernels for Celerite GP.
- Implement least bad outlier removal.
- Check that the mapping between (DeltaF, Fbase) and (FS, Fblend) is correct.
- Performance testing for sampling, ideally SBC. 
- Handling multi-wavelength observations.
- Plotting functions are a mess and they eat up way too much memory, there has to be a 
bus somewhere
- All PSPL models have a nested structure but I'm not sure how to utilize class
inheritance to construct nested models within PyMC3. 
Last time I tried this I ran into problems.
- Documentation via ReadTheDocs
- Installation via pip
