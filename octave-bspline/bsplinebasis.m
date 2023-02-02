% B = bsplinebasis(degree,knots,ts)
%
% Evaluate basis for a spline of given degree and knots, at a set of parameter
% values ts.
%
% Each column corresponds to a control point and each row corresponds to one
% parameter value such that, for a column vector of control points c, the
% spline can be evaluated as:
%
% s = B.c
%
% Matthew Chapman <contact@zmatt.net> and Eugenio Piasini
% <eugenio.piasini@gmail.com>, based on algorithm in C code for An
% Introduction to NURBS, David F. Rogers, Section 3.5

function B = bsplinebasis(degree,knots,ts)

% ensure ts is a column vector
ts = reshape(ts, [length(ts), 1]);

order = degree + 1;
nknots = length(knots);
npoints = nknots-order;

n_samples = length(ts);

% special case: if t is last knot, treat it as if it were in the previous
% span, otherwise it would not be in any
searcht = ts;
last_different_knot = find(knots~=knots(end),1,'last');
searcht(searcht==knots(end)) = knots(last_different_knot);

% calculate 1st order basis functions
% 1 if in knot span, 0 if not
temp = sparse(n_samples, nknots-1);
for j = 1:nknots-1
    temp(:,j) = double(searcht >= knots(j) & searcht < knots(j+1));
end

for k = 2:order
    % recursively calculate next order basis functions
    % by a linear combination of temp(j) and temp(j+1)
    for j = 1:nknots-k
        d = zeros(n_samples, 1);
        e = zeros(n_samples, 1);
        
        mask_d = temp(:,j)~=0;
        knotsdelta_d = knots(j+k-1)-knots(j);
        if knotsdelta_d~=0
            d(mask_d) = ((ts(mask_d)-knots(j)).*temp(mask_d,j))/knotsdelta_d;
        end

        mask_e = temp(:,j+1) ~= 0;
        knotsdelta_e = knots(j+k)-knots(j+1);
        if knotsdelta_e~=0
            e(mask_e) = ((knots(j+k)-ts(mask_e)).*temp(mask_e,j+1))/knotsdelta_e;
        end


        temp(:,j) = d+e;
    end
end

B = temp(:,1:npoints);

end
