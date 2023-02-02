classdef session < matlab.mixin.Copyable
    %SESSION Experimental session
    %   An experimental session contains multiple neurons, as well as the
    %   recording of all external and internal predictors

    properties (Constant)
        chunk_duration = 200 % duration (in ms) of the chunks for crossvalidation
    end
    
    properties
        experiment
        model
        neurons
        n_timesteps
        n_trials
        t_start
        cam_ts
        cam_ts_trial_id % trial number for each time step of the fluorescence recording
        cam_ts_chunk_id % partitioning of the fluorescence time steps into "chunks" for crossvalidation
        event_data
        trial_data
        head_velocity
        animal_id
        design_matrix
        fold_id
        pathway = ''
    end
    
    methods
        %% ----CORE LOADING METHOD----
        % This is where we conver the data to the format the rest of the
        % code expects, and it is where most of the complexity of 'session'
        % lies. It is also the first part of the code that will need to be
        % modified to use this library with another dataset.
        function obj = session(filename, pathway)
            %SESSION Construct a session instance from a datafile
            
            if nargin < 1
                return
            end
            if nargin < 2
                pathway = '';
            end
            % store which pathway was recorded in this session
            obj.pathway = pathway;
            
            % load data from disk
            data = load(filename, 'msdata_cleaned');
            data = data.msdata_cleaned;
            
            % extract basic info on data dimensions and animal
            signal = data.sigfn;
            [n_neurons, obj.n_timesteps] = size(signal);
            n_events = size(data.event_ts, 1);
            obj.animal_id = data.event_ts(1).animal_id;
            
            % create array of neurons recorded in this session
            obj.neurons = neuron.empty();
            for n = 1:n_neurons
                obj.neurons(n) = neuron(obj, signal(n,:)');
            end
            
            % extract fluorescence trace time steps and set reference frame
            % to first time step
            obj.t_start = data.mscam_ts(1);
            obj.cam_ts = data.mscam_ts' - obj.t_start;
            
            % create array of trial identities for each cam timestep
            obj.n_trials = max([data.event_ts.trial_num]);
            obj.cam_ts_trial_id = zeros(size(obj.cam_ts, 1), 1);
            trial_start = 1000*(data.SessionData.TrialStartTimestamp-data.SessionData.TrialStartTimestamp(1));
            for trial=1:obj.n_trials
                obj.cam_ts_trial_id(obj.cam_ts>=trial_start(trial)) = trial;
            end
            
            % create array of "chunk ids" which will be used for
            % stratifying the crossvalidation folds
            obj.cam_ts_chunk_id = 1+floor(obj.cam_ts/obj.chunk_duration);           
  
            % extract simple events (i.e., those related to external
            % predictors)
            obj.event_data = table(...
                [data.event_ts.event_ts]' - obj.t_start,...
                repmat("", n_events, 1),...
                ones(n_events, 1),...
                [data.event_ts.trial_num]',...
                'VariableNames', {'time', 'name', 'value',  'trial'});
            
            for e_id=1:n_events
                this_event_name = data.event_ts(e_id,1).event_name;
                if strcmp(this_event_name, 'reward')
                    % rename "reward" to "outcome" following what Kyu said
                    obj.event_data{e_id,'name'} = "outcome";
                else
                    obj.event_data{e_id,'name'} = string(this_event_name);
                end
            end
            
            % add table with information about events that do not just
            % happen, but can happen in more than one way. We call this the
            % value of the event (for instance choice could be left or
            % right).
            obj.trial_data = table(...
                [data.SessionData.Choice]',...
                [data.SessionData.Reward]',...
                cell2mat(data.SessionData.RewardVolume)',...
                'VariableNames', {'choice', 'outcome', 'reward_volume'});
            % force outcome=0 events to be read as outcome=-1
            outcome_0_trial_ids = obj.trial_data{:, 'outcome'} == 0;
            obj.trial_data{outcome_0_trial_ids, 'outcome'} = -1;
            
            for e_id=1:n_events
                name = obj.event_data{e_id,2};
                trial = obj.event_data{e_id,4};
                if name=="choice"
                    obj.event_data(e_id,:).value = obj.trial_data(trial,:).choice;
                elseif name=="outcome"
                    obj.event_data(e_id,:).value = obj.trial_data(trial,:).outcome;
                end
            end
            
            % add separate, dummy-coded predictors for events with values.
            % This allows for instance to model neurons that only respond
            % to left choice.
            predictors_to_be_dummy_coded = {'choice', 'outcome'};
            for p_id=1:length(predictors_to_be_dummy_coded)
                predictor_name = predictors_to_be_dummy_coded{p_id};
                values = unique(obj.trial_data{:,predictor_name});
                for v_id=1:length(values)
                    value = values(v_id);
                    new_predictor_name = sprintf("%s_%d", predictor_name, value);
                    orig_event_ids = strcmp(obj.event_data.name, predictor_name) & obj.event_data.value==value;
                    new_event_data = table(...
                        obj.event_data{orig_event_ids, 'time'},...
                        repmat(new_predictor_name, nnz(orig_event_ids), 1),...
                        ones(nnz(orig_event_ids),1),...
                        obj.event_data{orig_event_ids, 'trial'},...
                        'VariableNames', {'time', 'name', 'value',  'trial'});
                    obj.event_data = [obj.event_data; new_event_data];
                end
            end
            
            % add choice x outcome interaction
            choice_values = unique(obj.trial_data{:,'choice'});
            assert(length(choice_values)==2);
            outcome_values = unique(obj.trial_data{:,'outcome'});
            for choice_id = 1:length(choice_values)
                choice = choice_values(choice_id);
                other_choice = choice_values(3-choice_id);
                for outcome_id = 1:length(outcome_values)
                    outcome = outcome_values(outcome_id);
                    new_predictor_name = sprintf("choice_%d:outcome_%d", choice, outcome);
                    
                    this_choice_trials = find(obj.trial_data.choice==choice & obj.trial_data.outcome==outcome);
                    other_choice_trials = find(obj.trial_data.choice==other_choice & obj.trial_data.outcome==outcome);
                    
                    % contrast coding: weight 1/2 for choice/outcome event
                    % that match this predictor, weight -1/2 for the
                    % opposite choice
                    contrast_weight = 1/2;
                    
                    % add interaction events that match this predictor
                    orig_event_ids = strcmp(obj.event_data.name, "outcome") & ismember(obj.event_data.trial, this_choice_trials);
                    new_event_data = table(...
                        obj.event_data{orig_event_ids, 'time'},...
                        repmat(new_predictor_name, nnz(orig_event_ids), 1),...
                        contrast_weight * ones(nnz(orig_event_ids),1),...
                        obj.event_data{orig_event_ids, 'trial'},...
                        'VariableNames', {'time', 'name', 'value', 'trial'});
                    obj.event_data = [obj.event_data; new_event_data];
                    
                    % add interaction events with the same outcome but
                    % opposite choice
                    orig_event_ids = strcmp(obj.event_data.name, "outcome") & ismember(obj.event_data.trial, other_choice_trials);
                    new_event_data = table(...
                        obj.event_data{orig_event_ids, 'time'},...
                        repmat(new_predictor_name, nnz(orig_event_ids), 1),...
                        -contrast_weight * ones(nnz(orig_event_ids),1),...
                        obj.event_data{orig_event_ids, 'trial'},...
                        'VariableNames', {'time', 'name', 'value', 'trial'});
                    obj.event_data = [obj.event_data; new_event_data];                    
                end
            end

            if any(strcmp('head_velocity', fieldnames(data)))
                % add head velocity predictor, if present in the data. We
                % need to interpolate the original data array to match the
                % timestamps of the imaging data.
                %
                % Note that here we are also changing the scale of the head
                % velocity by multiplying it by 10^13. This gives head
                % velocity values of the order of 1, which are much nicer
                % to look at than whatever enormous scale was coming out of
                % the recordings.
                obj.head_velocity = interp1(data.behav_ts'-obj.t_start, ...
                    data.head_velocity' * 10^13, ...
                    obj.cam_ts, ...
                    "linear", "extrap");
            end
            
            tethering_events = ["init_start", "choice", "outcome"];
                
            if any(strcmp('softmaxResult', fieldnames(data)))
                % add events related to internal predictors, such as those that
                % come from the reinforcement learning model.
                
                % ensure that our notion of how many trials there are,
                % coming from the imagig data structure, matches the length
                % of the data structures used to store the results of the
                % RL model
                assert(size(data.softmaxResult.Qvalues,2)==obj.n_trials)
                
                % Q value, separately for left and right choice, tethered
                % to initiation, choice and outcome.
                for te_id=1:length(tethering_events)
                    tethering_event = tethering_events(te_id);
                    orig_event_ids = find(strcmp(obj.event_data.name, tethering_event));
                    n_events = length(orig_event_ids);
                    choice_values = unique(obj.event_data{strcmp("choice", obj.event_data.name), 'value'});
                    for c_id=1:length(choice_values)
                        choice_value = choice_values(c_id);
                        new_predictor_name = sprintf("Q_value_%d_T%s", choice_value, tethering_event);
                        qvalues = zeros(n_events,1);
                        for e_id=1:n_events
                            trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                            qvalues(e_id) = data.softmaxResult.Qvalues(c_id,trial);
                        end
                        new_event_data = table(...
                            obj.event_data{orig_event_ids, 'time'},...
                            repmat(new_predictor_name, nnz(orig_event_ids), 1),...
                            qvalues,...
                            obj.event_data{orig_event_ids, 'trial'},...
                            'VariableNames', {'time', 'name', 'value',  'trial'});
                        obj.event_data = [obj.event_data; new_event_data];
                    end
                end
                
                % Q value of the chosen option, tethered to initiation,
                % choice and outcome.
                for te_id=1:length(tethering_events)
                    tethering_event = tethering_events(te_id);
                    new_predictor_name = sprintf("Q_chosen_T%s", tethering_event);
                    orig_event_ids = find(strcmp(obj.event_data.name, tethering_event));
                    n_events = length(orig_event_ids);
                    qchosen = zeros(n_events, 1);
                    for e_id = 1:n_events
                        trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                        this_choice = data.softmaxResult.choices(trial);
                        qchosen(e_id) = data.softmaxResult.Qvalues(this_choice,e_id);
                    end
                    new_event_data = table(...
                        obj.event_data{orig_event_ids, 'time'},...
                        repmat(new_predictor_name, n_events, 1),...
                        qchosen,...
                        obj.event_data{orig_event_ids, 'trial'},...
                        'VariableNames', {'time', 'name', 'value',  'trial'});
                    obj.event_data = [obj.event_data; new_event_data];
                end
                
                % absolute Q difference (abs(QL-QR))
                new_predictor_name = "Q_difference";
                orig_event_ids = find(strcmp(obj.event_data.name, "init_start"));
                n_events = length(orig_event_ids);
                qdiffs = zeros(n_events, 1);
                for e_id=1:n_events
                     trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                     qdiffs(e_id) = data.softmaxResult.QDifferences(trial);
                end
                new_event_data = table(...
                    obj.event_data{orig_event_ids, 'time'},...
                    repmat(new_predictor_name, n_events, 1),...
                    qdiffs,...
                    obj.event_data{orig_event_ids, 'trial'},...
                    'VariableNames', {'time', 'name', 'value',  'trial'});
                obj.event_data = [obj.event_data; new_event_data];
                
                % Q sum (Q(+1)+Q(-1)), tethered to outcome
                new_predictor_name = "Q_sum";
                orig_event_ids = find(strcmp(obj.event_data.name, "outcome"));
                n_events = length(orig_event_ids);
                qsums = zeros(n_events, 1);
                for e_id=1:n_events
                     trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                     qsums(e_id) =  data.softmaxResult.Qvalues(1,trial)+data.softmaxResult.Qvalues(2,trial);
                end
                new_event_data = table(...
                    obj.event_data{orig_event_ids, 'time'},...
                    repmat(new_predictor_name, n_events, 1),...
                    qsums,...
                    obj.event_data{orig_event_ids, 'trial'},...
                    'VariableNames', {'time', 'name', 'value',  'trial'});
                obj.event_data = [obj.event_data; new_event_data];
                
                % Q signed difference (Q(+1)-Q(-1)), tethered to outcome           
                new_predictor_name = "Q_signed_difference_Toutcome";
                orig_event_ids = find(strcmp(obj.event_data.name, "outcome"));
                n_events = length(orig_event_ids);
                qdiffs = zeros(n_events, 1);
                for e_id = 1:n_events
                    trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                    qdiffs(e_id) = data.softmaxResult.Qvalues(2,trial)-data.softmaxResult.Qvalues(1,trial);
                end
                new_event_data = table(...
                    obj.event_data{orig_event_ids, 'time'},...
                    repmat(new_predictor_name, n_events, 1),...
                    qdiffs,...
                    obj.event_data{orig_event_ids, 'trial'},...
                    'VariableNames', {'time', 'name', 'value',  'trial'});
                obj.event_data = [obj.event_data; new_event_data];
                % Q signed difference (rectified)
                obj.add_rectified_predictor(new_predictor_name);
                
                % Q signed difference (Q(+1)-Q(-1)), tethered to init_start           
                new_predictor_name = "Q_signed_difference_Tinit_start";
                orig_event_ids = find(strcmp(obj.event_data.name, "init_start"));
                n_events = length(orig_event_ids);
                qdiffs = zeros(n_events, 1);
                for e_id=1:n_events
                    trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                    qdiffs(e_id) = data.softmaxResult.Qvalues(2,trial)-data.softmaxResult.Qvalues(1,trial);
                end
                new_event_data = table(...
                    obj.event_data{orig_event_ids, 'time'},...
                    repmat(new_predictor_name, n_events, 1),...
                    qdiffs,...
                    obj.event_data{orig_event_ids, 'trial'},...
                    'VariableNames', {'time', 'name', 'value',  'trial'});
                obj.event_data = [obj.event_data; new_event_data];
                % Q signed difference (rectified)
                obj.add_rectified_predictor(new_predictor_name);                
                
                % Reward Prediction Error
                new_predictor_name = "RPE";
                orig_event_ids = find(strcmp(obj.event_data.name, "outcome"));
                n_events = length(orig_event_ids);
                rpes = zeros(n_events, 1);
                for e_id = 1:n_events
                    trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                    this_choice = data.softmaxResult.choices(trial);
                    rpes(e_id) = data.softmaxResult.RPEs(this_choice,trial);
                end
                new_event_data = table(...
                    obj.event_data{orig_event_ids, 'time'},...
                    repmat(new_predictor_name, n_events, 1),...
                    rpes,...
                    obj.event_data{orig_event_ids, 'trial'},...
                    'VariableNames', {'time', 'name', 'value',  'trial'});
                obj.event_data = [obj.event_data; new_event_data];
                
                % Reward Prediction Error - rectified (split in positive
                % and negative components)
                obj.add_rectified_predictor(new_predictor_name);
                
            end
            
            % Compute and add state estimate from recursive logistic
            % regression model
            temp_data = {};
            temp_data.nTrials = [data.SessionData.nTrials];
            temp_data.t0choice = data.SessionData.Choice;
            temp_data.t0outcome = data.SessionData.Choice .* (data.SessionData.Reward>0);
            data_logreg = struct();
            data_logreg(1).recording_catenate = temp_data;
            [~, all_state, ~] = recursive_logistic_regression_behavior(data_logreg, false);
            all_state = all_state{1};
            for te_id=1:length(tethering_events)
                % recursive logistic regression state
                tethering_event = tethering_events(te_id);
                new_predictor_name = sprintf("state_T%s", tethering_event);
                orig_event_ids = find(strcmp(obj.event_data.name, tethering_event));
                n_events = length(orig_event_ids);
                states = zeros(n_events, 1);
                for e_id=1:n_events
                    trial = obj.get_trial_from_timestamp(obj.event_data{orig_event_ids(e_id), 'time'});
                    states(e_id) = all_state(trial);
                end
                new_event_data = table(...
                    obj.event_data{orig_event_ids, 'time'},...
                    repmat(new_predictor_name, nnz(orig_event_ids), 1),...
                    states,...
                    obj.event_data{orig_event_ids, 'trial'},...
                    'VariableNames', {'time', 'name', 'value',  'trial'});
                obj.event_data = [obj.event_data; new_event_data];
                
                % recursive logistic regression state (rectified)
                obj.add_rectified_predictor(new_predictor_name);
            end
            
        end
        
        %% ----PUBLIC COMPUTATION METHODS----
        % These are the methods that are meant to be employed directly by
        % the user to compute stuff (as opposed to plotting methods, that
        % live in their separate section below).
        function event_types = get_available_event_types(obj)
            event_types = ["autoregressive"; unique(obj.event_data.name)];
        end
        
        function dm = get_design_matrix(obj, config)
            %GET_DESIGN_MATRIX Compute or load design matrix
            if nargin == 1
                config = obj.model.design_matrix_config;
            end

            % Make sure this DM config is the DM config for the experiment
            % (this affects all other sessions too)
            obj.experiment.set_design_matrix(config);
            
            if isempty(obj.design_matrix)
                % if needed, actually build the matrix
                obj.design_matrix = obj.build_design_matrix();
            end
            dm = obj.design_matrix;
            n_columns_total = size(dm,2);
            
            % remove columns for excluded predictors
            excluded_columns_logical = false(1,n_columns_total);
            n_excluded_predictors = length(config.excluded_predictors);
            for p_id=1:n_excluded_predictors
                predictor_name = config.excluded_predictors(p_id);
                predictor_columns = config.column_indices(char(predictor_name));
                excluded_columns_logical(predictor_columns) = true;
            end
            dm = dm(:,~excluded_columns_logical);
            
        end
        
        function tuning = get_population_tuning(obj, predictors, dm_config)
           %GET_POPULATION_TUNING return SIMULTANEOUS population tuning to
           %given predictors.
           %
           %   This generalizes the notion of neuronal tuning to many
           %   neurons recorded simultaneously in this session. For this to
           %   work, the neurons need to all have the same set of "active
           %   trials".
            if nargin < 3
                dm_config = obj.model.design_matrix_config;
            end
            
            neurons_in_use = neuron.empty();
            for n=1:length(obj.neurons)
                this_neuron = obj.neurons(n);
                if ~this_neuron.excluded
                    neurons_in_use = [neurons_in_use, this_neuron];
                end
            end
            
            timesteps = obj.get_timestep_id_range([]);
            
            pop_trace = zeros(obj.n_timesteps, length(neurons_in_use));
            full_pop_pred = zeros(obj.n_timesteps, length(neurons_in_use));
            reduced_pop_pred =  zeros(obj.n_timesteps, length(neurons_in_use));
            for n=1:length(neurons_in_use)
                this_neuron = neurons_in_use(n);
                full_dm_config = this_neuron.get_full_dm_config(predictors, dm_config);
                pop_trace(:,n) = this_neuron.trace;
                full_pop_pred(:,n) = this_neuron.get_prediction(full_dm_config, timesteps);
                if this_neuron.is_trivial_reduced_model(predictors, full_dm_config)
                    reduced_pop_pred(:,n) = full_pop_pred(:,n);
                else
                    reduced_pop_pred(:,n) = this_neuron.get_prediction(full_dm_config.get_reduced_variant(predictors), timesteps);
                end
            end
            
            full_residuals = pop_trace - full_pop_pred;
            reduced_residuals = pop_trace - reduced_pop_pred;
            
            tot_var = scaled_generalized_variance(pop_trace);
            full_var = scaled_generalized_variance(full_residuals);
            reduced_var = scaled_generalized_variance(reduced_residuals);
            
            tuning = 100*(reduced_var - full_var)/tot_var;
            
        end
        
        function tuning = get_population_tuning_to_each_predictor(obj, dm_config)
            %GET_TUNING_TO_EACH_PREDICTOR return the tuning of this neuron
            %to all the active predictors, taken individually, and to the
            %special predictor groups, for the given DM config.
            if nargin < 2
                dm_config = obj.model.design_matrix_config;
            end
            
            predictor_names = dm_config.get_active_predictors();
            n_predictors = length(predictor_names);
            
            tuning = zeros(1, n_predictors);
            column_names = cell(1, n_predictors);
            
            % collect tuning indices to simple predictors
            for p_id=1:n_predictors
                predictor_name = predictor_names(p_id);
                tuning(p_id) = obj.get_population_tuning(predictor_name);
                % adjust the names of the predictor to make valid matlab
                % table column names.
                column_names{p_id} = obj.neurons(1).predname2varname(predictor_name);
            end
            if dm_config.use_special_predictor_groups
                % collect tuning indices to special predictor sets
                set_names = dm_config.special_predictor_sets.keys;
                for s_id=1:length(dm_config.special_predictor_sets)
                    tuning_ind_position = n_predictors+s_id;
                    set_name = set_names{s_id};
                    predictor_set = dm_config.get_special_predictor_set(set_name);
                    tuning(tuning_ind_position) = obj.get_population_tuning(predictor_set);
                    column_names{tuning_ind_position} = set_name;
                end
            end
            tuning = array2table(tuning, 'VariableNames', column_names);
        end
        
        function fit_all_models(obj, dm_config, parallel_workers)
            %FIT_ALL_MODELS fit all models (full and reduced) for all
            %neurons in the session.
            %
            %   See |get_deviances| for information on the
            %   |parallel_workers| argument.
            if nargin == 1 || isempty(dm_config)
                dm_config = obj.model.design_matrix_config;
            end
            if nargin < 3
                parallel_workers = [];
            end
            obj.get_deviances(dm_config, parallel_workers);
        end
        
        function [fde_full, fde_reduced] = get_deviances(obj, dm_config, parallel_workers)
            %GET_DEVIANCES Compute fraction of deviance explained for all
            %cells, for full and all reduced models.
            %
            %   If |parallel_workers| is empty, this will be run serially.
            %   If it is not empty and a parallel pool already exists, the
            %   existing pool will be used. If a pool does not exist, a
            %   pool of size |parallel_workers| will be created. If
            %   |parallel_workers| is 0, the default number of workers will
            %   be used.
            
            if nargin == 1 || isempty(dm_config)
                dm_config = obj.model.design_matrix_config;
            end
            if nargin < 3
                parallel_workers = [];
            end
            
            n_predictors = length(dm_config.get_active_predictors());
            fde_full = zeros(length(obj.neurons), 1);
            if dm_config.use_special_predictor_groups
                n_reduced_fdes = n_predictors+length(dm_config.special_predictor_sets);
            else
                n_reduced_fdes = n_predictors;
            end
            fde_reduced = zeros(length(obj.neurons), n_reduced_fdes);
            
            if isempty(parallel_workers)
                % run serially
                for n=1:length(obj.neurons)
                    [fde_full(n), fde_reduced(n,:)] = obj.neurons(n).get_full_and_reduced_fraction_deviance_explained(dm_config);
                end
            else
                % create a pool of parallel_workers workers if a pool is
                % not there yet, and parallelize over the neurons in the
                % session
                if isempty(gcp('nocreate'))
                    if parallel_workers==0
                        parpool('local');
                    else
                        parpool('local', parallel_workers);
                    end
                end
                
                % make sure that the design matrix has been already
                % computed, to avoid each parallel worker having to
                % re-compute it on its own.
                obj.get_design_matrix(dm_config);
                
                % in order to keep parfor happy, we have to explicitly
                % break out the session's neurons in an array, compute the
                % deviances, and explicitly reassign the neurons back. If
                % we don't do this, all the internal changes to the neuron
                % objects would be lost.
                
                % start by creating an empty array of neurons
                neurons_array = neuron.empty();
                % populate the array with "pruned" neurons, that is,
                % neurons such that their parent session is a session that
                % only has that neuron in the neuron list. This avoids an
                % issue where matlab would serialize a non-pruned neuron by
                % copying its session and, recursively, each of its neurons
                % with their sessions (or something like that), which would
                % cause the memory usage to balloon in the parfor below
                for n=1:length(obj.neurons)
                    neurons_array(n) = obj.get_pruned_neuron(n);
                end
                % run the fits for each pruned neuron in parallel
                parfor n=1:length(neurons_array)
                    this_neuron = neurons_array(n)
                    [fde_full(n), fde_reduced(n,:)] = this_neuron.get_full_and_reduced_fraction_deviance_explained(dm_config);
                    neurons_array(n) = this_neuron;
                end
                % once the fits are computed, overwrite the "fit" property
                % of each (original, unpruned) neuron with the fit property
                % of the correspoding pruned neuron.
                for n=1:length(obj.neurons)
                    obj.neurons(n).fit = neurons_array(n).fit;
                end
            end
            
        end
        
        function ts_range = get_timestep_id_range(obj, trial_range)
            %GET_TIMESTEP_ID_RANGE compute range of timestep indices from trial range
            %   If the trial range is empty, the full range of timestamps
            %   will be returned.
            if isempty(trial_range)
                ts_range = 1:length(obj.cam_ts);
            else
                ts_range = [];
                for t_id=1:length(trial_range)
                    trial = trial_range(t_id);
                    ts_range = [ts_range, find(obj.cam_ts_trial_id==trial)'];
                end
            end
        end
        
        function tuning = get_tuning(obj, dm_config)
            %GET_TUNING Compute tuning index to all individual predictors,
            %for all neurons.
            %
            %   |tuning| is an n_neurons x n_predictors table of tuning
            %   indices.
            
            if nargin < 2
                dm_config = obj.model.design_matrix_config;
            end
            
            tuning = table();
            
            for n=1:length(obj.neurons)
                tuning = [tuning; obj.neurons(n).get_tuning_to_each_predictor(dm_config)];
            end
        end
                
        function trial = get_trial_from_timestamp(obj, timestamp)
            %GET_TRIAL_FROM_TIMESTAMP get the trial index for a timestamp.
            closest_cam_ts = find(obj.cam_ts>=timestamp, 1, 'first');
            trial = obj.cam_ts_trial_id(closest_cam_ts);
        end
        
        
        function clear_design_matrix(obj)
            %CLEAR_DESIGN_MATRIX clear the stored design matrix, if present
            obj.design_matrix = [];
        end
        
        %% ----PLOTTING METHODS----
        % These are also public methods, all related to plotting things (as
        % opposed to computing things).
        function fh = plot_predictions(obj, dm_config)
            
            if nargin < 2
                dm_config = obj.model.design_matrix_config;
            end
            
            fh = figure('Name', 'Fluorescence traces with predictions');
            n_cols = 3;
            n_rows = ceil(length(obj.neurons)/n_cols);
            for n=1:length(obj.neurons)
                ah = subplot(n_rows, n_cols, n);
                obj.neurons(n).plot_prediction(dm_config, ah);
                text(0.1, 0.8, sprintf("Neuron %d", n), 'Units', 'normalized');
            end
            
            
        end
        
        
        function fh = plot_design_matrix(obj, trial_range, config)
            %PLOT_DESIGN_MATRIX Visualize design matrix for a set of trials
            
            if nargin<2 || isempty(trial_range)
                trial_range = 1:10;
            end
            
            if nargin<3
                config = obj.model.design_matrix_config;
            end
            
            ts_range = obj.get_timestep_id_range(trial_range);
            
            fh = figure('Position', [0,0,1024,640]);
            %fh = figure();
            dm = obj.get_design_matrix(config);
            dm = dm(ts_range,:);
            dm = dm./max(abs(dm),[],1);
            imagesc(dm);
            cbh = colorbar;
            ylabel(cbh, "Column-max-normalized predictor")
            
            predictor_names = config.get_active_predictors();
            predictor_label_locs = zeros(1,length(predictor_names));
 
            hold on;
            xl = xlim;
            yl = ylim;
            for p_id=1:length(predictor_names)
                name = char(predictor_names(p_id));
                cols = config.column_indices(name);
                predictor_label_locs(p_id) = mean(cols);
                if p_id>1
                    plot([cols(1), cols(1)]-0.5, yl, 'Color', 'k');
                end
            end
            xlim(xl);
            ylim(yl);
            xticks(predictor_label_locs);
            xticklabels(arrayfun(@(s) strrep(s,"_"," "), predictor_names));
            set(gca,'XAxisLocation','top')
            
            % label trials
            trial_label_locs = zeros(1,length(trial_range));
            row_counter = 1;
            for t_id=1:length(trial_range)
                trial = trial_range(t_id);
                rows = row_counter:row_counter+nnz(obj.cam_ts_trial_id==trial);
                row_counter = rows(end);
                trial_label_locs(t_id) = mean(rows);
                if t_id>1
                    plot(xl, [rows(1), rows(1)]-0.5, 'Color', 'r');
                end
            end
            xlim(xl);
            ylim(yl);            
            yticks(trial_label_locs);
            yticklabels(arrayfun(@(n) sprintf("Trial %d", n), trial_range));
        end
        
        
        function [S,AX,BigAx,H,HAx] = plot_dm_column_pairs(obj, predictor_names, dm_config)
            %PLOT_DM_COLUMN_PAIRS pairplot for the columns of the design
            %matrix involving the specified predictors.
            %
            %   |predictor_names| is an array of strings.
            %
            %   This is useful to run sanity checks on the distribution of
            %   the elements of the design matrix, and to check for
            %   collinearity.
            %
            %   Keep in mind that this is pretty heavy computationally, so
            %   try to keep the number of predictors as small as possible.
            if nargin<3
                dm_config = obj.model.design_matrix_config;
            end
            
            dm = obj.get_design_matrix(dm_config);
            
            selected_columns = [];
            label_positions = zeros(1,length(predictor_names));
            for p_id=1:length(predictor_names)
                predictor = predictor_names(p_id);
                this_pred_columns = obj.model.design_matrix_config.column_indices(char(predictor));
                
                label_positions(p_id) = length(selected_columns) + length(this_pred_columns)/2;
                
                selected_columns = [selected_columns, this_pred_columns];
            end
            
            % generate main plots
            [S,AX,BigAx,H,HAx] = plotmatrix(dm(:,selected_columns));
            
            % place predictor name labels
            for p_id=1:length(predictor_names)
                predictor = predictor_names(p_id);
                text(BigAx,...
                    label_positions(p_id)/length(selected_columns),...
                    1.02,...
                    strrep(predictor, "_", " "),...
                    'Units', 'normalized');
            end
        end
        
        
        function fh1 = plot_deviance_summary(obj, dm_config)
            
            if nargin<2
                dm_config = obj.model.design_matrix_config;
            end
 
            predictor_names = [dm_config.get_active_predictors(), string(dm_config.special_predictor_sets.keys)];
            n_predictors = length(predictor_names);
            n_neurons = length(obj.neurons);
            
            fh1 = figure('Name', 'Fraction of explained deviance for full model and reduced models (percent)',...
                'Position', [0,0,1024,640]);
            for n=1:n_neurons
                ah = subplot(n_neurons, 1, n);
                obj.neurons(n).plot_deviance_summary(dm_config, ah);
                ylabel({"FDE (%)", sprintf("Neuron %d", n)})
                if n==n_neurons
                    xticks(0:n_predictors);
                    xticklabels(["Full", arrayfun(@(s) strrep(s,"_"," "), predictor_names)]);
                else
                    xticks([]);
                    xticklabels([]);
                end
            end
        end
        
        function fh = plot_tuning_summary(obj, dm_config)
            
            if nargin<2
                dm_config = obj.model.design_matrix_config;
            end
            
            [fde_full, ~] = obj.get_deviances(dm_config);
            
            fh = figure('Name', 'Tuning summary',...
                'Position', [0,0,780,600]);
            hold on
            n_rows = 1 + ceil(sqrt(length(obj.neurons)));
            n_cols = ceil(sqrt(length(obj.neurons)));

            % plot individual spider plots
            for n=1:length(obj.neurons)
                ph = subplot(n_rows,n_cols,n_cols+n);
                obj.neurons(n).plot_tuning_summary(dm_config, ph);
                title(sprintf("Neuron %d", n));
            end            
            
            % add summary of full FDE on top
            subplot(n_rows,1,1)
            bar(1:length(obj.neurons),100*fde_full);
            xlabel("Neuron ID")
            ylabel("FDE (%, full model)")
            
        end

        
        function [kernel_f, axh] = plot_kernels(obj, dm_config)
            %PLOT_KERNELS Plot all fitted kernels for all neurons
            
            if nargin<2
                dm_config = obj.model.design_matrix_config;
            end
            
            predictor_names = dm_config.get_active_predictors();
            n_predictors = length(predictor_names);
                
            kernel_f = figure('Name', 'Fitted kernels', 'Position', [0,0,1024,640]);
            axh = zeros(length(obj.neurons), n_predictors);
            kernel_scales = zeros(length(obj.neurons), 2);
            kernel_scales_internal = zeros(length(obj.neurons), 2);
            kernel_scales_reward = zeros(length(obj.neurons), 2);
            kernel_scales_headvelocity = zeros(length(obj.neurons), 2);
            
            for n=1:length(obj.neurons)
                neuron = obj.neurons(n);
                [kernels, lags] = neuron.get_kernels(dm_config);
                
                for i=1:n_predictors
                    predictor_name = char(predictor_names(i));
                    axh(n,i) = subplot(length(obj.neurons), n_predictors, (n-1)*n_predictors+i);
                    hold on
                    area(lags(predictor_name), kernels(predictor_name));
                    
                    if ismember(predictor_name, obj.model.design_matrix_config.special_predictor_sets('internal'))
                        kernel_scales_internal(n,1) = min(kernel_scales_internal(n,1), min(kernels(predictor_name)));
                        kernel_scales_internal(n,2) = max(kernel_scales_internal(n,2), max(kernels(predictor_name)));
                    elseif strcmp(predictor_name, "reward_rate")
                        kernel_scales_reward(n,1) = min(kernel_scales_reward(n,1), min(kernels(predictor_name)));
                        kernel_scales_reward(n,2) = max(kernel_scales_reward(n,2), max(kernels(predictor_name)));
                    elseif strcmp(predictor_name, "head_velocity")
                        kernel_scales_headvelocity(n,1) = min(kernel_scales_headvelocity(n,1), min(kernels(predictor_name)));
                        kernel_scales_headvelocity(n,2) = max(kernel_scales_headvelocity(n,2), max(kernels(predictor_name)));                        
                    else
                        kernel_scales(n,1) = min(kernel_scales(n,1), min(kernels(predictor_name)));
                        kernel_scales(n,2) = max(kernel_scales(n,2), max(kernels(predictor_name)));
                    end
                    if n<length(obj.neurons)
                        set(gca,'xtick',[])
                        set(gca,'xticklabel',[])
                    end
                    if n==1
                        title(strrep(predictor_name, '_', ' '));
                    elseif n==length(obj.neurons)
                        xlabel('Time (ms)')
                    end
                end
                
                if all(kernel_scales(n,:)==0)
                    kernel_scales(n,:) = [-1, 1];
                end
                if all(kernel_scales_internal(n,:)==0)
                    kernel_scales_internal(n,:) = [-1, 1];
                end
                if all(kernel_scales_reward(n,:)==0)
                    kernel_scales_reward(n,:) = [-1, 1];
                end
                if all(kernel_scales_headvelocity(n,:)==0)
                    kernel_scales_headvelocity(n,:) = [-1, 1];
                end
                
                for i=1:n_predictors
                    predictor_name = char(predictor_names(i));
                    if ismember(predictor_name, obj.model.design_matrix_config.special_predictor_sets('internal'))
                        set(axh(n,i), 'YLim', 1.2*kernel_scales_internal(n,:));
                    elseif strcmp(predictor_name, "reward_rate")
                        set(axh(n,i), 'YLim', 1.2*kernel_scales_reward(n,:));
                    elseif strcmp(predictor_name, "head_velocity")
                        set(axh(n,i), 'YLim', 1.2*kernel_scales_headvelocity(n,:));                        
                    else
                        set(axh(n,i), 'YLim', 1.2*kernel_scales(n,:));
                    end
                end
                
                for i=1:n_predictors
                    subplot(length(obj.neurons), n_predictors, (n-1)*n_predictors+i);
                    xl = xlim;
                    yl = ylim;
                    gray = [0.6, 0.6, 0.6];
                    plot([0,0], yl, 'LineStyle', ':', 'Color', gray)
                    plot(xl, [0,0], 'LineStyle', ':', 'Color', gray)
                    ylim(yl)
                    xlim(xl) 
                end
            end
        end
        
    end
    
    %% ----PRIVATE INTERFACE----
    % These are methods that are only supposed to be used by other
    % components of the library.
    methods (Access=?neuron)
        function fold_id = get_folds(obj)
            %GET_FOLDS Generate stratification vector for glmnet
            %    The idea here is that we divide the dataset in many small
            %    "chunks" of duration obj.cam_ts_chunk_id, and we assign
            %    each of them randomly to one of the crossvalidation folds.
            
            n_folds = obj.model.glm_fit_config.n_folds;
            
            if isempty(obj.fold_id) || max(unique(obj.fold_id))~=n_folds
                random_trial_assignment = randi(n_folds, [obj.cam_ts_chunk_id(end),1]);
                obj.fold_id = random_trial_assignment(obj.cam_ts_chunk_id);
            end
            fold_id = obj.fold_id;
        end 
    end
    
    methods (Access=?experiment)
        function pruned_session = prune(obj, neuron_id)
            %PRUNE return "pruned" session object, where all neurons except
            %the requested one have been swapped with empty dummies.
            
            pruned_session = copy(obj);
            
            for n=1:length(obj.neurons)
                if n~=neuron_id
                    pruned_session.neurons(n) = neuron([], []);
                else
                    pruned_session.neurons(n) = copy(obj.neurons(n));
                    pruned_session.neurons(n).session = pruned_session;
                end
            end
        end
    end
       
    methods (Access=private)
        
        function dm = build_design_matrix(obj)
            %BUILD_DESIGN_MATRIX Generate a new design matrix
            dm = [];
            for predictor_idx = 1:height(obj.model.design_matrix_config.config)
                this_event_name = obj.model.design_matrix_config.config(predictor_idx,1).Row{1,1};
                % check that the requested event type is available (i.e.,
                % check that it has data associated with it).
                if (strcmp(this_event_name, 'head_velocity') && ~any(strcmp('head_velocity', fieldnames(obj)))) ||...
                    (~any(strcmp(this_event_name, obj.get_available_event_types())) && ~any(strcmp(this_event_name, ["autoregressive", "reward_rate", "head_velocity"])))
                    warning("the design matrix configuration included event type %s, but no data is present for that event type. Please check the event type name.", this_event_name);
                end
                
                n_knots = obj.model.design_matrix_config.config(this_event_name,:).n_knots;
                if strcmp(this_event_name, "autoregressive")
                    % this is a special autoregressive predictor,
                    % representing time-shifted activity of a neuron. and
                    % as such it does not need time expansion. Here we just
                    % insert one or more placeholder columns (depending on
                    % the order of the term) in the design matrix, which
                    % each neuron will fill in independently before fitting
                    % the model.
                    this_dm = sparse(length(obj.cam_ts), n_knots);
                elseif strcmp(this_event_name, "head_velocity")
                    % this is a special continuous predictor encoding the
                    % head velocity
                    this_dm = sparse(obj.head_velocity);
                elseif strcmp(this_event_name, "reward_rate")
                    % this is a special continuous predictor which stores
                    % the running average of the reward rate in μl/min over
                    % the last |n_knots| trials.
                    this_dm = zeros(length(obj.cam_ts), 1);
                    prev_trial_end = 0;
                    prev_trial_ts_id = 0;
                    for trial=1:obj.n_trials
                        window_start_trial = max(trial-n_knots, 1);
                        window_start_ts = obj.get_timestep_id_range(window_start_trial);
                        window_start_ts = obj.cam_ts(window_start_ts(1));
                        base_reward_sum = sum(obj.trial_data{window_start_trial:trial-1,'reward_volume'});
                        if trial > 1
                            base_delta_t = prev_trial_end - window_start_ts + 1;
                        else
                            base_delta_t = 0;
                        end
                        ts = obj.cam_ts(obj.get_timestep_id_range(trial));
                        outcome_ts = obj.event_data{find(obj.event_data.trial==trial & strcmp(obj.event_data.name, "outcome"), 1),'time'};
                        reward_volume = obj.trial_data{trial, 'reward_volume'};
                        for t_id=1:length(ts)
                            t = ts(t_id);
                            if obj.trial_data{trial, 'reward_volume'}==0 || t<outcome_ts
                                reward_sum = base_reward_sum;
                            else
                                reward_sum = base_reward_sum + reward_volume;
                            end
                            delta_t = base_delta_t + (t - ts(1)) + 1;
                            this_dm(prev_trial_ts_id + t_id) = (60*1000) * (reward_sum / delta_t); % (conversion to μl/min)
                        end
                        prev_trial_end = ts(end);
                        prev_trial_ts_id = prev_trial_ts_id + length(ts);
                    end
                elseif n_knots==0
                    % this is a stepwise predictor, which takes on the
                    % value of its associated event, holding it until the
                    % event itself and switching to the value of the next
                    % event of the same type immediately after.
                    this_dm = zeros(length(obj.cam_ts), 1);
                    this_event_t = double(obj.event_data{obj.event_data.name==string(this_event_name),'time'});
                    this_event_values = obj.event_data{obj.event_data.name==string(this_event_name),'value'};
                    prev_cam_ts = 1;
                    for e_idx = 1:length(this_event_t)
                        event = this_event_t(e_idx);
                        closest_cam_ts = find(obj.cam_ts>=event, 1, 'first');
                        value = this_event_values(e_idx);
                        this_dm(prev_cam_ts:closest_cam_ts-1) = value;
                        prev_cam_ts = closest_cam_ts;
                    end
                    this_dm(prev_cam_ts:end) = value;
                    this_dm = sparse(this_dm);
                else
                    % this is a regular kernel-based predictor.
                    degree = obj.model.design_matrix_config.config(this_event_name,:).degree;
                    n_kernels = n_knots + 2 + degree - 1;
                    this_dm = sparse(length(obj.cam_ts), n_kernels);
                    this_event_t = double(obj.event_data{obj.event_data.name==string(this_event_name),'time'});
                    this_event_values = obj.event_data{obj.event_data.name==string(this_event_name),'value'};
                    
                    event_type_knots = obj.model.get_default_knots(obj.model.design_matrix_config.config(this_event_name,:));
                    
                    for e_idx = 1:length(this_event_t)
                        event = this_event_t(e_idx);
                        value = this_event_values(e_idx);
                        event_knots = event_type_knots + event;
                        this_dm = this_dm + value * bsplinebasis(degree, event_knots, obj.cam_ts);
                    end
                end
                dm = [dm, this_dm];
            end
            
            % lock ("freeze") the design matrix configuration to make sure
            % it will not be modified in the future, making it inconsistent
            % with this DM.
            obj.model.design_matrix_config.frozen = true;
        end
        
        
        function add_rectified_predictor(obj, predictor_name)
            %ADD_RECTIFIED_PREDICTOR generate and add to the event table
            %two new predictors representing the positive and negative part
            %of the given valued predictor.
            %
            % Note that the predictor will be "rectified" in the sense that
            % the sign of the negative part will be flipped. So the end
            % result will be two new positive-valued predictors, one equal
            % to the positive part of the original predictor and one equal
            % to minus its negative part.
            
            [new_event_data_pos, new_event_data_neg] = obj.get_rectified_event_tables(predictor_name);
            obj.event_data = [obj.event_data; new_event_data_pos; new_event_data_neg];
        end
        
        function [event_data_pos, event_data_neg] = get_rectified_event_tables(obj, predictor_name)
            %GET_RECTIFIED_EVENT_TABLES build new event tables for the
            %positive part and minus the negative part of a given valued
            %event.
            
            orig_event_ids = strcmp(obj.event_data.name, predictor_name);
            
            % extract events where event value is >= 0
            new_event_ids = orig_event_ids & obj.event_data.value>=0;
            new_predictor_name = sprintf("%s_pos", predictor_name);
            event_data_pos = table(...
                obj.event_data{new_event_ids, 'time'},...
                repmat(new_predictor_name, nnz(new_event_ids), 1),...
                obj.event_data{new_event_ids, 'value'},...
                obj.event_data{new_event_ids, 'trial'},...
                'VariableNames', {'time', 'name', 'value',  'trial'});
            
            % extract events where event value is negative
            new_event_ids = orig_event_ids & obj.event_data.value<0;
            new_predictor_name = sprintf("%s_neg", predictor_name);
            event_data_neg = table(...
                obj.event_data{new_event_ids, 'time'},...
                repmat(new_predictor_name, nnz(new_event_ids), 1),...
                -obj.event_data{new_event_ids, 'value'},...
                obj.event_data{new_event_ids, 'trial'},...
                'VariableNames', {'time', 'name', 'value',  'trial'});

        end
        
        function pruned_neuron = get_pruned_neuron(obj, neuron_id)
            %GET_PRUNED_NEURON get "pruned" neuron object, containing only
            %the data needed to fit a given neuron.
            pruned_session = obj.prune(neuron_id);
            pruned_neuron = pruned_session.neurons(neuron_id);
        end
        
        
    end
    
end
