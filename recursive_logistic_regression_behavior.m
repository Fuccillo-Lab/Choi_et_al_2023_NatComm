% This model predicts the probability of
% the animal choosing right. The model has two predictors:
%
% 1. past choice. This predictor has values +1 for right and -1 for left.
%
% 2. whether the previous trial was rewarded (that is, if it was a "win"), and
% if it was, which side it was. This predictor has value +1 if the previous
% choice was right and it got a reward, -1 if the previous choice was left
% and it got a reward, and 0 if it was unrewarded.
%
% 3. past estimate of latent state (in practice this is computed from the
% past choices and rewards).

function [b, s, lo] = recursive_logistic_regression_behavior(data, make_plots)
% |data| is expected to be of the format used by Kyu for the file
% "sessions_miniscope.mat". If the variable contained in that file is
% called |animal_list|, you can call this function with something like
% |recursive_logistic_regression_behavior(animal_list, 8)| to analyze the
% animals' behavior by using up to 8 trials in the past to build the model
% predictors.
%
% make_plots controls if a figure will be generated. By default it is true,
% but you can set it to false if you just want the fitted coefficient for
% your own plotting.
%
% Returns: b is a table with n_mice rows and 4 column, containing the
% fitted values of the parameters and the deviance for each individual
% mouse.

if nargin < 2
    make_plots = false;
end

n_mice = length(data);
b = table();
for mouse=1:n_mice
    this_b = recursive_logistic_regression_single_mouse(data(mouse).recording_catenate);
    b = [b; this_b];
end



s = {};
lo = {};
for mouse=1:n_mice
    [this_s, this_lo] = get_recursive_regression_state(b{mouse,:}, data(mouse).recording_catenate);
    s = [s; this_s];
    lo = [lo; this_lo];
end


if make_plots
    
    
    for mouse=1:n_mice
        
        mouse_data = data(mouse).recording_catenate;
        
        [state, log_odds] = get_recursive_regression_state(b{mouse,:}, mouse_data);
        
        choice = mouse_data.t0choice';
        choice_l = find(choice==-1);
        choice_r = find(choice==1);
        
        figure();
        subplot(1,2,1)
        hold on      
        plot(log_odds)
        scatter(choice_r, log_odds(choice_r), 'MarkerFaceColor', 'r');
        scatter(choice_l, log_odds(choice_l), 'MarkerFaceColor', 'y');
        title('log odds')
        subplot(1,2,2)
        hold on
        plot(state)
        scatter(choice_r, state(choice_r), 'MarkerFaceColor', 'r');
        scatter(choice_l, state(choice_l), 'MarkerFaceColor', 'y');     
        title('state')
        
    end
end
end

function [sol, deviance] = recursive_logistic_regression_single_mouse(mouse_data)

opts = optimset('MaxFunEvals', 10000);
[sol, deviance] = fminsearch(@(params) cost_function(params, mouse_data), [0, 1, 0, 1], opts);
deviance_se = sqrt(2*variance_of_log_likelihood(sol, mouse_data));

rate_choose_right = mean(mouse_data.t0choice==1);
null_deviance = -2 * (nnz(mouse_data.t0choice==1) * log(rate_choose_right) + nnz(mouse_data.t0choice==-1) * log(1-rate_choose_right));
fde = (null_deviance-deviance)/null_deviance;
n_params = length(sol);
aic = deviance + 2*n_params;


sol = array2table([sol, deviance, deviance_se, fde, aic], 'VariableNames', {'alpha', 'beta', 'delta', 'tau', 'Deviance', 'Deviance_SE', 'FDE', 'AIC'});

end


function cost = cost_function(params, mouse_data)


choice = mouse_data.t0choice';

[~, log_odds] = get_recursive_regression_state(params, mouse_data);

likelihood = sigmoid(log_odds);

% convert choice to 0/1
choice(choice==-1) = 0;

log_likelihood = choice .* log(likelihood) + (1-choice) .* log(1-likelihood);

cost = -2 * sum(log_likelihood);

%lo = ln p(r)/p(l)
%p(r) = 1/(1+exp(-lo));
%ln p(r) = -ln(1+exp(-lo))


%[b, dev] = glmfit(dm{:,:}, choice, 'binomial');
%b = array2table([b', dev], 'VariableNames', [{'Intercept'}, dm.Properties.VariableNames, {'Deviance'}]);

end

function var = variance_of_log_likelihood(params, mouse_data)

[~, log_odds] = get_recursive_regression_state(params, mouse_data);
likelihood = sigmoid(log_odds);

var = sum(likelihood.*(1-likelihood).*log_odds.^2);

end

function h = sigmoid(x)

h = 1 ./ (1 + exp(-x));

end