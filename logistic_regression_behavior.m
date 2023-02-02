% Simple logistic regression model for mouse behavior.
%
% The general assumption is that we want to show that the animal pays
% attention to the history of rewards, and not just to what its previous
% choices were. In order to do this, this model predicts the probability of
% the animal choosing right. The model has two predictors:
%
% 1. past choice. This predictor has values +1 for right and -1 for left.
%
% 2. whether the previous trial was rewarded (that is, if it was a "win"), and
% if it was, which side it was. This predictor has value +1 if the previous
% choice was right and it got a reward, -1 if the previous choice was left
% and it got a reward, and 0 if it was unrewarded.
%
% Note that these definitions are given by taking the right side as the
% reference, but because everything is symmetric you can also interpret
% predictor 1 as capturing the tendency of the mice to stay with their
% previous choice in unrewarded trials, and predictor 2 as capturing the
% "pull" of a reward, on either side.

function b = logistic_regression_behavior(data, n_lags, make_plots)
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
% Returns: b is a table with n_mice rows and n_predictor columns,
% containing the fitted values of the parameters for the logistic
% regression for each individual mouse. The names of the table's columns
% give the name of the parameters. For example, |choicem2| means "the
% choice of the animal, two trials back" ("m2" stands for "minus two").

if nargin < 2
    n_lags = 5;
end

if nargin < 3
    make_plots = true;
end

n_mice = length(data);
b = table();
for mouse=1:n_mice
    b = [b; logistic_regression_single_mouse(data(mouse).recording_catenate, n_lags)];   
end

if make_plots
    
    var_names = {'choice', 'win'};
    
    for v_id=1:length(var_names)
        this_var = var_names{v_id};
        this_columns = startsWith(b.Properties.VariableNames, this_var);
        this_params = b{:,this_columns};
        
        subplot(1,length(var_names),v_id);
        hold on
        title(this_var);
        
        % plot individual mice
        for mouse=1:n_mice
            h = plot(1:n_lags, this_params(mouse,:), 'Marker', 'o');
            set(h, 'MarkerFaceColor', get(h,'Color'));
        end
        
        % plot population average
        plot(1:n_lags, mean(this_params), 'Color', 'k', 'LineWidth', 3);
        
        
        xlim([0.5, n_lags+0.5])
        ylim([-2,2])
        xlabel('Lag (trials)')
        if v_id==1
            ylabel('Weight')
        end
        
        xl = xlim;
        h = plot(xl, [0,0], 'LineStyle', ':', 'LineWidth', 0.5, 'Color', 'k');
        xlim(xl);
        uistack(h,'bottom')
        
    end
end
end

function [b, dev] = logistic_regression_single_mouse(mouse_data, n_lags)

dm = table();
choice = [];

n_sessions = length(mouse_data.nTrials);

trial_counter = 1;
for s=1:n_sessions
    session_trials = trial_counter:trial_counter+mouse_data.nTrials(s)-1;
    session_choice = mouse_data.t0choice(session_trials)';
    session_outcome = mouse_data.t0outcome(session_trials)';
    dm = [dm; build_dm(session_choice, session_outcome, n_lags)];
    choice = [choice; mouse_data.t0choice(trial_counter+n_lags:trial_counter+mouse_data.nTrials(s)-1)'];
    trial_counter = trial_counter + mouse_data.nTrials(s);
end

choice(choice==-1) = 0;
[b, dev] = glmfit(dm{:,:}, choice, 'binomial');

log_odds = [ones(height(dm),1), table2array(dm)] * b;
likelihood = 1./(1+exp(-log_odds));
dev_var = sum(likelihood.*(1-likelihood).*log_odds.^2);


rate_choose_right = mean(mouse_data.t0choice==1);
null_deviance = -2 * (nnz(mouse_data.t0choice==1) * log(rate_choose_right) + nnz(mouse_data.t0choice==-1) * log(1-rate_choose_right));
fde = (null_deviance-dev)/null_deviance;
aic = dev + (1 + 2*n_lags);

b = array2table([b', dev, sqrt(dev_var), fde, aic], 'VariableNames', [{'Intercept'}, dm.Properties.VariableNames, {'Deviance', 'Deviance_SE', 'FDE', 'AIC'}]);

end

function var = variance_of_log_likelihood(params, mouse_data)

[~, log_odds] = get_recursive_regression_state(params, mouse_data);
likelihood = sigmoid(log_odds);

var = sum(likelihood.*(1-likelihood).*log_odds.^2);

end

function dm = build_dm(session_choice, session_outcome, n_lags)

dm = table();

for lag=1:n_lags
    this_choice = add_lag(session_choice, lag);
    this_outcome = add_lag(session_outcome, lag);
    this_outcome(this_outcome==-1) = 0;
    this_win = this_outcome;
    this_win(this_choice==-1) = -1;
    dm = [dm table(this_choice, this_win,...
        'VariableNames',...
        {sprintf('choicem%d', lag), sprintf('winm%d', lag)})];    
end
% remove first trials (as we don't have info on lagged choice and outcome
% for them)
dm = dm(n_lags+1:end,:);
end
    
function lagged_A = add_lag(A, n_lags)

lagged_A = circshift(A, n_lags);
lagged_A(1:n_lags) = NaN;


end