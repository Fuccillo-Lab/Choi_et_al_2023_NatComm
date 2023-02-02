function [state, log_odds] = get_recursive_regression_state(params, mouse_data)
%GET_RECURSIVE_REGRESSION_STATE compute the latent "state" phi for the
%recursive logistic regression model of behavior as defined by Beron,
%Neufeld, Linderman and Sabatini 2022.
%
%   Arguments:
%
%   |params| is an array containing the fitted parameters of the recursive
%   logistic regression model
%
%   |mouse_data| is formatted like the "recording_catenate" field  in the
%   data structure used to fit the full logistic or recursive logistic
%   regression models.

alpha = params(1);
beta = params(2);
delta = params(3);
tau = params(4);

n_sessions = length(mouse_data.nTrials);
total_trials = sum(mouse_data.nTrials);

state = zeros(total_trials, 1);
log_odds = zeros(total_trials ,1);

choice = mouse_data.t0choice';
outcome = mouse_data.t0outcome';

trial_counter = 1;
for s=1:n_sessions
    n_trials = mouse_data.nTrials(s);
    for t=trial_counter:trial_counter+n_trials-1
        if t>trial_counter
            state(t) = beta * outcome(t) + exp(-1/tau) * state(t-1);
        end
    end
    log_odds(trial_counter+1:trial_counter+n_trials-1) = delta + alpha*choice(trial_counter:trial_counter+n_trials-2) + state(trial_counter:trial_counter+n_trials-2);
    trial_counter = trial_counter + n_trials;
end

end