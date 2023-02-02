function v = scaled_generalized_variance(X)
%SCALED_GENERALIZED_VARIANCE a measure of scatter scale for data X
%
%   X is n_samples x n_variables.
%   
%   See Paindaveine 2008, doi:10.1016/j.spl.2008.01.094 for why this
%   particular choice has nice statistical properties.


k = size(X,2);
sigma = cov(X);
v = nthroot(det(sigma), k);

end