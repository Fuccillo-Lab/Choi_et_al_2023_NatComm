% This model predicts the probability of
% the animal choosing right. The model has two predictors:
%
% 1. past choice. This predictor has values +1 for right and -1 for left.
%
% 2. whether the previous trial was rewarded (that is, if it was a "win"), and
% if it was, which side it was. This predictor has value +1 if the previous
% choice was right and it got a reward, -1 if the previous choice was left
% and it got a reward, and 0 if it was unrewarded.

function b = noisy_wstlsw_behavior(data)
% |data| is expected to be of the format used by Kyu for the file
% "sessions_miniscope.mat". If the variable contained in that file is
% called |animal_list|, you can call this function with something like
% |logistic_regression_behavior(animal_list, 8)| to analyze the animals'
% behavior by using up to 8 trials in the past to build the model
% predictors.
%
% make_plots controls if a figure will be generated. By default it is true,
% but you can set it to false if you just want the fitted coefficient for
% your own plotting.
%
% Returns: b is a table with n_mice rows and 1 column, containing the
% fitted values of the lapse rate for each individual mouse.

% if nargin < 2
%     make_plots = false;
% end

n_mice = length(data);
b = table();
for mouse=1:n_mice
    b = [b; recursive_logistic_regression_single_mouse(data(mouse).recording_catenate)];   
end

end

function [sol, deviance] = recursive_logistic_regression_single_mouse(mouse_data)

opts = optimset('MaxFunEvals', 10000);
[sol, deviance] = fminsearch(@(params) cost_function(params, mouse_data), 0.1, opts);

deviance_se = sqrt(2*variance_of_log_likelihood(sol, mouse_data));

rate_choose_right = mean(mouse_data.t0choice==1);
null_deviance = -2 * (nnz(mouse_data.t0choice==1) * log(rate_choose_right) + nnz(mouse_data.t0choice==-1) * log(1-rate_choose_right));
fde = (null_deviance-deviance)/null_deviance;
aic = deviance + 1;

sol = array2table([sol, deviance, deviance_se, fde, aic], 'VariableNames', {'lapse_rate', 'Deviance', 'Deviance_SE', 'FDE', 'AIC'});

end

function var = variance_of_log_likelihood(params, mouse_data)

log_likelihood = get_wsls_log_likelihood(params, mouse_data);
likelihood = exp(log_likelihood);
log_odds = log(likelihood./(1-likelihood));

var = sum(likelihood.*(1-likelihood).*log_odds.^2);

end

function log_likelihood = get_wsls_log_likelihood(params, mouse_data)

lapse_rate = params(1);

n_sessions = length(mouse_data.nTrials);
total_trials = sum(mouse_data.nTrials);

choice = mouse_data.t0choice';
outcome = mouse_data.t0outcome';
log_likelihood = zeros(total_trials, 1) + log(1/2);


% convert choice to 0/1
choice(choice==-1) = 0;

% convert outcome to rewarded/not rewarded
rewarded = outcome ~= 0;

trial_counter = 1;
for s=1:n_sessions
    n_trials = mouse_data.nTrials(s);
    for t=trial_counter:trial_counter+n_trials-1
        if t>trial_counter
            
            stay = choice(t) == choice(t-1);
            
            if rewarded(t-1)
                log_likelihood(t) = stay * log(1-lapse_rate) + (1-stay) * log(lapse_rate);
            else
                log_likelihood(t) = stay * log(lapse_rate) + (1-stay) * log(1-lapse_rate);
            end
        end
    end
end
end



function cost = cost_function(params, mouse_data)

cost = -2 * sum(get_wsls_log_likelihood(params, mouse_data));

end