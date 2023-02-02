%% Example analysis flow

%% Summary
% This is an example analysis script, meant to show how to use the code in
% this repo. Please see the README file for software requirements and for
% the expected data format.
%
% This script can be expected to run in a few minutes (under 10) on a
% 4-core laptop.

%% Load data from disk
% Again, see the README for information about how the file should be
% structured internally.
%
% Just for the sake of example, we tag session 1 as belonging to the aDMS
% pathway and session 2 as belonging to pDMS. This is not necessarily the
% case in reality.

s1 = session('example_session_data_1.mat', 'aDMS');
s2 = session('example_session_data_2.mat', 'pDMS');

%% Configure design matrix
% Configure how we will build the design matrix, namely which events we
% want to include as predictors, how many knots we want to use for the
% splines, and how large the kernel windows should be.
%
% The number of knots controls how flexible our splines are. Here for
% convenience we fix this to be the same for all predictors, but note that
% this parameter is still specified independently for each predictor when
% we actually configure the design matrix below.

n_knots = 3;

%%%
% Create the configuration object. We can add predictors using the
% |add_predictor| method, and by specifying the name of the predictor, the
% number of spline knots, and the duration of the kernel window before and
% after the event (in milliseconds). If a predictor is requested that
% doesn't have any corresponding event in the data, a warning is raised.
% This is the function signature for adding predictors:

% config.add_predictor(predictor_name, window_pre, window_post, n_knots)

% Note that session 2 doesn't have the RL model outputs, so we omit those
% for the time being.

%% Add external predictors
%
% Remember that you can always get a list of available predictors (i.e.,
% predictors that can be added as we do here) by using
% session.get_available_event_types()

dm_config = designMatrixConfiguration();
dm_config.add_predictor('center_on', n_knots, 0, 4000);
dm_config.add_predictor('init_start', n_knots, 300, 3000);

%dm_config.add_predictor('init_done', n_knots, 3000, 3000);
dm_config.add_predictor('choice_-1', n_knots, 1500, 1500);
dm_config.add_predictor('choice_1', n_knots, 1500, 1500);

% outcome predictors - "unsigned" outcome predictor with a pre-outcome
% window to capture anticipatory activity, plus outcome-specific predictors
% with a post-outcome window to capture neurons that respond to either
% positive or negative outcome.
dm_config.add_predictor('outcome_-1', n_knots, 0, 7000);
dm_config.add_predictor('outcome_1', n_knots, 0, 7000);

% Add interaction terms. Remember that
% 
% * only this specific interaction (choice x outcome) is supported
% * you can only add interactions between predictors (main effects) that
% are already present (so if you add choice_1:outcome_1 before adding
% choice_1, and error will be raised)
% * the window/kernel settings for choice x outcome should match those of
% outcome
% * for any value of outcome, you only need to add ONE interaction term.
% For instance, if you add |choice_1:outcome_1| then this predictor will
% represent the ADDITIONAL response to outcome=1 in trials where choice was
% 1 in comparison to those where choice was -1.
dm_config.add_predictor('choice_1:outcome_-1', n_knots, 0, 7000);
dm_config.add_predictor('choice_1:outcome_1', n_knots, 0, 7000);

%% Add head velocity predictor
% This predictor doesn't have any option: it is continuous and simply
% tracks the recorded head velocity, linearly interpolated in
% correspondence of each imaging timestamp.
dm_config.add_predictor('head_velocity')

%% Add internal predictors
% Internal predictors can be added using commands like those below. Here we
% can't actually add most of these as session 2 doesn't contain the
% information from the RL model needed to use any of the internal
% predictors except the reward rate.

% Add predictors for Q sum and Q difference. Here we set n_knots=0 to
% signify that these predictors should be represented "stepwise" and not
% with kernels. In other words, these predictors are simple constants that
% change value at the time of occurrence of their associated event (here,
% the outcome). If by mistake we included some values for window_pre and
% window_post, these would be ignored.
dm_config.add_predictor('Q_sum', 0);
dm_config.add_predictor('Q_signed_difference_Toutcome', 0);

% % Add predictor for environment state estimate by recursive regression
% % model. Here we used the "signed" version (that is, the version of the
% % predictor that takes on positive and negative values, as opposed to the
% % versions of the predictor that only encode the positive or negative part
% % of the state), tethered to outcome. Remember that typically you want to
% % add EITHER state OR the RL predictors (Q values etc).
% dm_config.add_predictor('state_Toutcome', 0);

% % Add predictors for positive and negative part of RPE as separate
% % predictors (note that this still generates nonzero elements of the design
% % matrix in the trials where outcome=0).
% dm_config.add_predictor('RPE_pos', n_knots, 0, 4000);
% dm_config.add_predictor('RPE_neg', n_knots, 0, 4000);

% Add reward rate predictor. This is of the form
% add_predictor('reward_rate',-n_trials), where n_trials is the number of
% trials in the past you want to use to determine the running average of
% the reward rate.
dm_config.add_predictor('reward_rate', -5);

%% Configure regression model
% Fix the settings for how the model is fitted (in glmnet) and evaluated. I
% have set up sensible defaults here, so there is no need to actually
% modify the settings for the moment.

fit_config = glmFitConfiguration();
fit_config.method = 'glmnet_matlab';

%% Put everything together
% Combine data, design matrix settings and model fit settings into one
% top-level |experiment| object. The first argument to experiment() is an
% array of sessions - here we only pass one because in our sample data the
% second session doesn't have the RL model data, so we wouldn't be able to
% use internal predictors if we kept both sessions.

e = experiment([s1, s2], dm_config, fit_config);

%% Specify subset of neurons and trials we want the model to look at
% For instance, say that we want to exclude completely neuron 4 in session
% 1, and limit the trials considered for neuron 1 in session 2 to trials 40
% through 120. Note that, like with all other options, this will remain
% stored within the neuron itself (under the |excluded| and |active_trials|
% properties), so if you save the experiment object to disk and you reload
% it later you can always figure out what was the range of active trials
% you used to fit the models for each neuron.

e.sessions(1).neurons(4).exclude_from_analysis();
e.sessions(1).neurons(2).set_active_trials(40:120);
e.sessions(1).neurons(4).set_active_trials(40:120);

%% Visualize design matrix for example trials (22 to 27)
% In this plot, time increases as we go *down* along the y axis. In the
% full design matix, there is one row per imaging timestep. On the x axis,
% each column gives the values of a predictor. The predictors are grouped
% according to the type of event they represent. For instance, the columns
% labeled "Choice -1" contain the temporal expansion of the event "Choice"
% when the value of the choice is -1 ("choose left"). The number of columns
% (i.e. spline basis elements) for each event type is determined by the
% number of knots and the degree of the splines used for that event type.
% In our case, as we always use cubic splines (so the degree is fixe to 3),
% the number of columns/actual predictors representing an event turns out
% to be n_knots+4.
%
% Note also that, for visualization purposes, each column in this plot has
% been normalized by its maximum, so the values you see here are not
% exactly the values that go in the model. Anyway, each column is
% individually standardized when the fit is performed, so this doesn't
% really matter too much.

e.sessions(1).plot_design_matrix(22:35);

%% Fit all full and reduced models, parallelizing over neurons within each session
% To use whatever number of workers is the default on the current local
% profile, we can set |parallel_workers| to 0. Alternatively, we can set
% it to the number of desired workers, or set parallel_workers=[] to run
% the fits serially.
parallel_workers = 0;
e.fit_all_models(parallel_workers)

%% Fit model and visualize kernels for each neuron and each predictor
% In the figures, columns are predictors, rows are neurons.
%
% Under the hood, fits are performed the first time that they are needed -
% in this case, here we ask to see the fitted kernels, so the fits are
% performed in order to extract those. After they have been performed, they
% get stored internally, so the same fit is never re-computed twice. In
% practice, this means that calling this same method a second time would be
% much faster.

e.plot_kernels();


%% Visualize tuning properties
% The tuning index for a given predictor is defined as the difference of
% the fraction of deviance explained under the full model and under a
% reduced model that doesn't have that predictor (things are slightly more
% complicated for interactions).
%
% The first figure just plots all this information in detail (i.e. full and
% reduced FDE for each neuron and each predictor), while the second figure
% tries to summarize it in a more interpretable way.

e.plot_deviance_summary();

%%%
% The second figure then plots the tuning index for each neuron and each
% predictor in the form of a spider or star plot. In these plots, each
% predictor is associated with an "arm" of the star, and the length of that
% arm is proportional to the tuning of the neuron for that predictor. At
% the top of the same figure we also see again the total FDE for each
% neuron, for reference.

e.plot_tuning_summary();

%% Access detailed tuning metrics
% I have included methods to facilitate access to the tuning metrics at the
% neuron, session and experiment level. Here are a couple of examples.

%%%
% Get joint tuning of a neuron to an arbitrary set of predictors (will most
% likely require re-fitting at least the reduced model, which will happen
% automatically under the hood):
example_session_n = 1;
example_neuron_n = 6;
fprintf("Joint tuning of neuron %d in session %d to both choices is %f",...
    example_neuron_n,...
    example_session_n,...
    e.sessions(1).neurons(6).get_tuning_index(["choice_1", "choice_-1"]))

%%%
% Tabulate tuning of all neurons to each individual predictor in the
% experiment. Note that this can also be done at the single-session level
% by using the identically-named |get_tuning| method of |session|.
e.get_tuning()

%% Compute and plot pathway-level tuning
% Here we compute the global tuning of the aDMS and pDMS pathways. Note
% that if we have more than one session per pathway, these sessions are be
% merged into appropriate pseudo-populations in order to compute the
% pathway tuning.
e.plot_pathway_tuning()

%%%
% Now that we have seen the total tuning of each pathway, we look at the
% tendency of individual neurons to be "purely" or "sparsely" tuned to one
% predictor only, or to multiple predictors. This is captured by the
% "population sparseness" metric, which is summarized in the following
% plot.
e.plot_pathway_tuning_sparseness()

% We can also plot the tuning sparseness for a subset of neurons above a
% certain threshold for the total tuning. Here we do that for a threshold
% of 5% FVE. Note that we need to pass 'false' here as the first argument,
% because that argument is only needed to control whether group predictors
% should be included in the sparseness calculation (which in general is not
% what you want).
e.plot_pathway_tuning_sparseness(false, 0.05)

%%%
% This can be further studied by looking specifically at the tuning for
% choice and choice x outcome on a neuron by neuron basis, and contrasted
% across pathways. In this plot, we also display a 2% tuning threshold for
% ease of interpretation.
e.plot_pathway_choice_outcome_mixed_tuning(2)

%%%
% We can now compare the temporal dynamics of the modulation of the two
% pathways by looking at their mean-displacement kernels. We also extract
% the p-values for each kernel and each lag, as well as the corresponding
% significance (true=significant, false=not significant) at the 0.05
% threshold.
[~, tests, pvalues] = e.plot_pathway_population_kernels(parallel_workers);

%%%
% We can also extract the population kernels for one of the populations for
% further processing. Here we show how to do that without computing the
% confidence intervals (which would otherwise be re-computed on the spot),
% if we're not interested in them.
[pop_kernels, ~, pop_kernels_lags] = e.populations('aDMS').get_population_kernels(e.model.design_matrix_config, [], [], false);
% pop_kernels and pop_kernels_lags are Maps indexed by the predictor names.
% So, for instance, we could make a simple plot of the right choice kernel
% by doing plot(pop_kernel_lags('choice_1'),pop_kernels('choice_1')).

%%%
% Finally, we show how to compute and plot the pathway kernels only for a
% subset of the neurons in each pathway (here we use neurons 1 to 10 in
% aDMS and 3 to 9 in pDMS. Note that in this case we have to specify all
% optional arguments to the |plot_pathway_population_kernels| function.
e.plot_pathway_population_kernels(parallel_workers, 0.05, e.model.design_matrix_config, 1:10, 3:9);


%%%
% It is possible to select subsets of neurons automatically, based on a
% simultaneous threshold on total FVE and tuning to a specific predictor.
% So for instance, we can select the neurons in aDMS with total tuning of
% at least 3% and all_outcome tuning of at least 1% as follows:
aDMS_neuron_subset_outcome = e.populations('aDMS').get_neurons_with_tuning_criterion(3, 'all_outcome', 1);

% In the same way, it is possible to filter neurons by total tuning only
% (here the threshold is 3%):
aDMS_neuron_subset_total = e.populations('aDMS').get_neurons_with_tuning_criterion(3);
pDMS_neuron_subset_total = e.populations('pDMS').get_neurons_with_tuning_criterion(3);

% These subsets can be given as argument to the functions discussed above,
% to extract or plot population kernels based only on these neuron subsets.
% For instance, we can plot population kernels for only the "good" neurons
% like so:
fh = e.plot_pathway_population_kernels(parallel_workers, 0.05, e.model.design_matrix_config, aDMS_neuron_subset_total, pDMS_neuron_subset_total);
set(fh, 'name', "Population kernels - tuned neurons only");

% Or we could extract the aDMS kernels (and their CIs) for a subset of
% neurons in one population like so:
[pop_kernels, pop_kernels_ci, pop_kernels_lags] = e.populations('aDMS').get_population_kernels(e.model.design_matrix_config, [], aDMS_neuron_subset_outcome);

% Finally, we could also just extract the kernel associated with a
% particular predictor. Note that in this case the outputs of the function
% are just simple arrays, not maps.
[outcome_1_kernel, outcome_1_kernel_ci, outcome_1_kernel_lags] = e.populations('aDMS').get_individual_population_kernel('outcome_1', e.model.design_matrix_config, [], aDMS_neuron_subset_outcome);


%% Other plots
% I have included facilities for making simple plots illustrating
% individual neurons more in detail. For instance, we can plot the activity
% and model fit for the recorded neurons, as well as the fitted kernels for
% an individual neuron.

e.plot_predictions();
e.sessions(1).neurons(1).plot_kernels();