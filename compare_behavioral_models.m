function [fde_RL, fde_logreg, fde_rec, fde_wl,...
    AIC_RL, AIC_logreg, AIC_rec, AIC_wl,...
    BIC_RL, BIC_logreg, BIC_rec, BIC_wl,...
    dev_se_RL, dev_se_logreg, dev_se_rec, dev_se_wl] = compare_behavioral_models(SessionData, n_lags)
%COMPARE_BEHAVIORAL_MODELS Compare goodness of fit of reinforcement
%learning, logistic regression models (full and recursive), and noisy
%win-stay/lose-switch.
% 
%   Arguments:
%
%   |SessionData| is in the usual format for Kyu's experiments, and the
%   same format that is accepted by Luigim's fitQModel_2CSR function.
%
%   |n_lags| is the number of lags to be included in the logistic
%   regression model. Typically 5.
%
%
%   Returns:
%
%   fde_{RL,logreg,rec,wl}: the fraction of deviance explained by each model.
%
%   AIC_{RL,logreg,rec,wl}: Akaike Information Criterion for each model. Lower is
%   better. Using this number to get a sense of the relative predictive
%   power of the two models is a good, conservative choice.
%
%   BIC_{RL,logreg,rec,wl}: Bayesian Information Criterion for each model. Lower
%   is better. This can also be used to compare the two models, but by
%   design it will be a bit less conservative in the sense that it will be
%   sloghtly more favorable towards the RL model by design (as it penalizes
%   more heavily models with larger numbers of parameters, and the RL
%   model has fewer parameters than the logistic regression one).
%
%   dev_se_{LR,logreg,rec,wl}: an estimate of the standard error of the
%   deviance for each of the model. This estimate is made under the
%   assumption that the considered model is correct.


%% fit RL model
fitted_model = fitQModel_2CSR(SessionData,'SoftMax');

[fde_RL, dev_RL, dev_null, dev_se_RL] = fraction_deviance_explained(fitted_model);
n_params_RL = 2;
AIC_RL = dev_RL + n_params_RL; % -2 log-likelihood + num_params
BIC_RL = dev_RL/2 + (n_params_RL/2)*log(SessionData.nTrials/(2*pi));

%% fit logistic regression model.
% here we have to rearrange the data a bit, because originally the code for
% the logistic regression was designed to fit one model only to multiple
% sessions for a given mouse. In order to compare properly to the RL model,
% we have to fit instead one logistic regression per session.

temp_data = {};
temp_data.nTrials = [SessionData.nTrials];
temp_data.t0choice = SessionData.Choice;
temp_data.t0outcome = SessionData.Choice .* (SessionData.Reward>0);

data_logreg = struct();
data_logreg(1).recording_catenate = temp_data;

b = logistic_regression_behavior(data_logreg, n_lags, false);
dev_logreg = b.Deviance;
fde_logreg = (dev_null - dev_logreg)/dev_null;
n_params_logreg = 1+2*n_lags;
AIC_logreg = dev_logreg + 2*n_params_logreg;
BIC_logreg = dev_logreg/2 + (n_params_logreg/2)*log(SessionData.nTrials/(2*pi));
dev_se_logreg = b.Deviance_SE;


%% fit recursive regression model.
[b, ~, ~] = recursive_logistic_regression_behavior(data_logreg, false);
dev_rec = b.Deviance;
fde_rec = (dev_null - dev_rec)/dev_null;
n_params_rec = 4;
AIC_rec = dev_rec + 2*n_params_rec;
BIC_rec = dev_rec/2 + (n_params_rec/2)*log(SessionData.nTrials/(2*pi));
dev_se_rec = b.Deviance_SE;


%% fit noisy win-stay/lose-switch model.
b = noisy_wstlsw_behavior(data_logreg);
dev_wl = b.Deviance;
fde_wl = (dev_null - dev_wl)/dev_null;
n_params_wl = 1;
AIC_wl = dev_wl + 2*n_params_wl;
BIC_wl = dev_wl/2 + (n_params_wl/2)*log(SessionData.nTrials/(2*pi));
dev_se_wl = b.Deviance_SE;


end




function [fde, dev, null_dev, stderr_dev] = fraction_deviance_explained(fitted_model)
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

% expected fde and its variance
p = fitted_model.choiceProbabilities(2,:);
var_dev = 2 * sum(p.*(1-p).*log(p./(1-p)).^2);
stderr_dev = sqrt(var_dev);


end