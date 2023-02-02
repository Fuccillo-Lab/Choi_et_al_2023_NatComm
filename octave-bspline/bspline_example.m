%% B-spline example
%
% This is a simple example illustrating the creation of a B-spline basis,
% the definition of a particular spline as a linear combination of the
% basis elements, and the computation of its derivatives.

%% main parameters for example
degree = 3; % degree of the spline basis
knots = [zeros(1,degree), linspace(0,1,7), ones(1,degree)]; % knot vector
xrange = -0.1:0.01:1.1; % range of values over which we will evaluate the splines

%% define basis functions and plot them
basis = bsplinebasis(degree, knots, xrange);

clf;
subplot(1,2,1);
hold on
title(sprintf("B-spline basis (degree %d, %d interior knots)", degree, length(knots)-2*(degree+1)));

yl = ylim();
for k=1:length(knots)
    knot = knots(k);
    plot([knot, knot], [-2,2], 'LineWidth', 0.5, 'Color', [0.6,0.6,0.6], 'LineStyle', ':');
end
plot(xrange, basis)
ylim([0,1]);

%% plot a random example linear combination of the basis elements and its first two derivatives
cpts = rand(size(basis,2),1)-0.5;
[d1cpts, d1knots, d1degree] = bsplinederiv(degree, cpts, knots);
[d2cpts, d2knots, d2degree] = bsplinederiv(d1degree, d1cpts, d1knots);

subplot(4,2,2);
plot(xrange, bsplineeval(degree, cpts, knots, xrange));
title("Random spline")
subplot(4,2,4);
bar(cpts);
title("Coefficients of random spline in b-spline basis");
subplot(4,2,6);
plot(xrange, bsplineeval(d1degree, d1cpts, d1knots, xrange));
title("First derivative")
subplot(4,2,8);
plot(xrange, bsplineeval(d2degree, d2cpts, d2knots, xrange));
title("Second derivative")
