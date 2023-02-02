function [state_rl, state_logreg, r, p, AIC_RL, AIC_rec] = compare_behavioral_state_estimates(SessionData)
%COMPARE_BEHAVIORAL_STATE_ESTIMATES Compare the state estimate you get out
%of the reinforcement learning model and the recursive logistic regression
%model.
% 
%   Arguments:
%
%   |SessionData| is in the usual format for Kyu's experiments, and the
%   same format that is accepted by Luigim's fitQModel_2CSR function.
%
%   Outputs:
%
%   state_{rl,rec}: the trial-by-trial array of inferred state, computed
%   with the two models (RL and recursive logistic regression)
%
%   r, p: pearson correlation coefficient and p-value for the correlation
%   between the two state estimates across trials
%
%   AIC_RL: Akaike Information Criterion for the RL model (lower is better)
%
%   AIC_rec: Akaike Information Criterion for the recursive logistic
%   regression model (lower is better)


%% fit RL model
fitted_model = fitQModel_2CSR(SessionData,'SoftMax');

[fde_RL, dev_RL, dev_null] = fraction_deviance_explained(fitted_model);
n_params_RL = 2;
AIC_RL = dev_RL + n_params_RL; % -2 log-likelihood + num_params
BIC_RL = dev_RL/2 + (n_params_RL/2)*log(SessionData.nTrials/(2*pi));

state_rl = fitted_model.Qvalues(2,:)-fitted_model.Qvalues(1,:);


%% prepare data for recursive logistic fit
temp_data = {};
temp_data.nTrials = [SessionData.nTrials];
temp_data.t0choice = SessionData.Choice;
temp_data.t0outcome = SessionData.Choice .* (SessionData.Reward>0);

data_logreg = struct();
data_logreg(1).recording_catenate = temp_data;


%% fit recursive regression model.
[b, s, lo] = recursive_logistic_regression_behavior(data_logreg, false);
dev_rec = b.Deviance;
fde_rec = (dev_null - dev_rec)/dev_null;
n_params_rec = 4;
AIC_rec = dev_rec + n_params_rec;
BIC_rec = dev_rec/2 + (n_params_rec/2)*log(SessionData.nTrials/(2*pi));

state_logreg = s{1}';
state_logreg_lo = lo{1}';

figure;
scatter(state_rl, state_logreg, 'filled');
xlabel('Q right - Q left')
ylabel('Recursive logistic regression state')

[r,p] = corrcoef([state_rl', state_logreg']);

text(0.15, 0.85, sprintf('r=%.3f\nAIC RL=%.3f\nAIC logreg=%.3f', r(1,2), AIC_RL, AIC_rec), 'Units', 'normalized')

end




function [fde, dev, null_dev, expected_fde, error_fde] = fraction_deviance_explained(fitted_model)
%% Compute fraction of deviance explained
% originally, choice is encoded as 1/2. Transform it to 0/1
choice = fitted_model.choices > 1;
% compute average probability of choice=1 ("p hat")
null_phat = mean(choice);
% compute log likelihood of null model, which always assigns the same
% probability to each choice
null_ll = sum(choice*log(null_phat) + (1-choice)*log(1-null_phat));
% compute null deviance (not really required but we'll do that for
% completeness)
null_dev = -2 * null_ll;

% compute RL model deviance. Note that the original code says that it
% returns "a measure of the likelihood of the model. Higher is better", but
% actually the returned measure is minus the log likelihood, and lower is
% better. Accordingly, the fitting code fits the model with fmincon, which
% minimizes its objective.
ll = -fitted_model.likelihood;
dev = -2 * ll;

% fraction of deviance explained
fde = (null_dev-dev)/null_dev;

% % expected fde and its variance
% p = fitted_model.choiceProbabilities(2,:);
% exp_dev = -2 * sum(p .* log(p));
% var_dev = sum(p.*(1-p).*log(p./(1-p)).^2);
% expected_fde = (null_dev - exp_dev)/null_dev;
% error_fde = sqrt(var_dev)/null_dev;


end