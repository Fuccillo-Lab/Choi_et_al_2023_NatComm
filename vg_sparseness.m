function s = vg_sparseness(x, dim)
%SPARSENESS Sparseness measure as in Vinje and Gallant 2000
if nargin < 2
    dim = find(size(x)>1, 1);
end
K = size(x, dim);
s = (K-(sum(x, dim).^2)./sum(x.^2, dim))/(K-1);
s(~any(s,dim)) = NaN;
end

