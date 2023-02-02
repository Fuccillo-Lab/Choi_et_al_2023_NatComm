classdef neuron < matlab.mixin.Copyable
    %NEURON An individual recorded neuron
    %   A neuron belongs to a session, has a fluorescence trace, and has
    %   methods to fit predictive models of the fluorescence trace, to
    %   extract the fitted kernels and to generate predictions based on the
    %   model. Under the hood, it stores the fits of the various variants
    %   of the encoding model that it computes, so that the same fit is
    %   never performed more than once.
    
    properties
        session % parent session
        trace % fluorescence trace
        excluded = false % whether this neuron is to be excluded from all analyses
    end
    properties (SetAccess=private)
        active_trials % trials that we intend to use to fit and evaluate the models
    end
    properties (SetAccess=?session)
        fit % container for model fits
    end
    
    %% ----PUBLIC INTERFACE----
    % These methods can be used to create a neuron object, to perform fits,
    % and to compute tuning indices.
    methods
        function obj = neuron(session, trace)
            %NEURON Define a recorded neuron from a fluorescence trace
            %
            %   trace : n_timepoints x 1 array of fluorescence values.
            %
            %   session : the neuron's recording session
            obj.session = session;
            obj.trace = trace;
            obj.fit = containers.Map;
            if ~isempty(session)
                obj.active_trials = 1:obj.session.n_trials;
            else
                obj.active_trials = [];
            end
        end
        
        function exclude_from_analysis(obj, desired_flag_state)
            %EXCLUDE_FROM_ANALYSIS modify 'excluded' flag that controls
            %inclusion of neuron in all modeling and analyses
            %
            %   By default, |desired_flag_state| can be omitted, and this
            %   will result in the neuron being exluded from all analyses.
            %   If a boolean value is passed as |desired_flag_state|, the
            %   |excluded| flag will be set to that state. This can be used
            %   to invert the exclusion.
            %
            %   When a neuron is excluded, it is skipped when performing 
            %   session-level analyses, and appropriate 'null' results are
            %   returned when neuron-level analysis methods are called.

            if nargin < 2
                desired_flag_state = true;
            end
            obj.excluded = desired_flag_state;
            obj.clear_fits();
        end
        
        function set_active_trials(obj, trial_range)
            %SET_ACTIVE_TRIALS specify the trials to be used by the models.
            %
            %   trial_range : an array of trial indices.
            %
            %   Note that all other trials will be excluded from model
            %   fitting and evaluation. Most notably, this is NOT meant to
            %   be a way of defining a validation set for the model, but
            %   rather just a way of specifying what range of trials we
            %   believe the recording to be "good" and worth considering.
            %
            %   Note that every time the trial range is changed, any
            %   existing model fit gets cleared from the cache, as it may
            %   now be invalid.
            
            if ~isequal(obj.active_trials, trial_range)
                obj.active_trials = trial_range;
                obj.clear_fits();
            end
        end
        
        
        function prediction = get_prediction(obj, dm_config, timesteps)
            %GET_PREDICTION get predicted activity of neuron during session
            
            if nargin < 3
                timestep_id_range = obj.get_timestep_id_range();
            else
                timestep_id_range = timesteps;
            end
            
            if obj.excluded
                prediction = NaN(length(timestep_id_range),1);
            else
                dm = obj.get_design_matrix(dm_config);
                this_fit = obj.get_fit(dm_config);
                
                if startsWith(obj.session.model.glm_fit_config.method, 'glmnet')
                    prediction = cvglmnetPredict(...
                        this_fit,...
                        dm(timestep_id_range,:),...
                        obj.session.model.glm_fit_config.lambda_selection_criterion,...
                        'response');
                elseif strcmp(obj.session.model.glm_fit_config.method, 'gglasso')
                    coefs = obj.get_coefs(dm_config);
                    prediction = coefs(1) + dm(timestep_id_range,:)*coefs(2:end);
                end
            end
        end
        
                
        function fde = get_fraction_deviance_explained(obj, dm_config)
            %GET_FRACTION_DEVIANCE_EXPLAINED return the FDE for this neuron
            %under the given model.
            
            if obj.excluded
                fde = NaN;
            else
                if nargin < 2
                    dm_config = obj.session.model.design_matrix_config;
                end
                
                this_fit = obj.get_fit(dm_config);
                fde = this_fit.glmnet_fit.dev(this_fit.glmnet_fit.lambda==this_fit.(obj.session.model.glm_fit_config.lambda_selection_criterion));
            end
        end
        
        function dm_config = get_full_dm_config(obj, predictors, dm_config)
            %GET_FULL_DM return the "full" DM config for the purpose of
            %measuring the tuning of the given predictor
            %
            %   decide what to use as a full model: if the requested
            %   predictors group contains a main effect in an interaction
            %   but not the interaction, and if the interaction has the
            %   same window as the main effect, this is the original full
            %   model minus the interaction term. Otherwise (i.e., if the
            %   group does not contain any main effects of interactions, or
            %   if these are not true main effects because their window
            %   doesn't match up with that of the interaction), we just use
            %   the full model
            %
            %   this is meant to capture the case where we ask for the
            %   tuning for 'outcome', but the full model contains both
            %   outcome and choice:outcome. Then the tuning should be
            %   determined by the difference in goodness of fit between the
            %   model without choice:outcome term and the model without
            %   both outcome and choice:outcome. Additionally, when
            %   computing the tuning for choice we should just get the
            %   standard comparison between full model and reduced model
            %   with no choice, because choice and choice:outcome have
            %   different windows.
            for p_id=1:length(predictors)
                predictor = predictors(p_id);
                related_interactions = dm_config.get_interactions_from_main_effect(predictor);
                
                if ~obj.is_trivial_reduced_model(related_interactions, dm_config)
                    for i_id=1:length(related_interactions)
                        interaction = related_interactions(i_id);
                        if ~ismember(interaction, predictors)
                            % remove this interaction from the full model.
                            % This is only done if (1) the interaction
                            % hasn't been removed already, and (2a) the
                            % interaction's fitted parameters in the full
                            % model are not all zero, or (2b) our settings
                            % say we should always fit reduced models
                            % regardless of whether some predictors have
                            % zero weight in the full model.
                            dm_config = dm_config.get_reduced_variant(interaction);
                        end
                    end
                end
            end
%             %DEBUG
%             fprintf("FULLDMCONFIG: Asked to compute full DM for %s, related interactions are %s\n",...
%                 strjoin(predictors), strjoin(related_interactions))
        end
        
        function tuning = get_tuning_index(obj, predictors, dm_config)
            %GET_TUNING_INDEX return the tuning index for the given group
            %of predictors. This is the COLLECTIVE tuning to all given
            %preditors SIMULTANEOUSLY.
            %
            %   |predictors| is a string or a array of strings

            if nargin < 3
                dm_config = obj.session.model.design_matrix_config;
            end
            
            % make sure predictors are specified as strings and not chars
            if ischar(predictors) || iscell(predictors)
                predictors = string(predictors);
            end
            
%             %DEBUG
%             fprintf("\nTUNING: Asked to compute tuning to %s\n",...
%                 strjoin(predictors));
            
            dm_config = obj.get_full_dm_config(predictors, dm_config);
            
%             %DEBUG
%             fprintf("TUNING: full model excludes predictors: %s\n",...
%                 strjoin(dm_config.excluded_predictors));
            
            fde_full =  obj.get_fraction_deviance_explained(dm_config);
            
            
            if obj.is_trivial_reduced_model(predictors, dm_config)
                % if the fitted coefficients for the given predictors are
                % all zero, and if |skip_zero_kernel_fde| is true, don't
                % bother fitting the reduced model and just assumed that
                % the reduced FDE will be the same as the full FDE
                fde_reduced = fde_full;
%                 %DEBUG
%                 fprintf("TUNING: no need to fit reduced model.\n");
            else
                % if this predictor has at least one nonzero
                % coefficient, build the reduced model and compute its
                % FDE
                dm_config_reduced = dm_config.get_reduced_variant(predictors);
%                 %DEBUG
%                 fprintf("TUNING: going to fit reduced model with excluded predictors: %s\n",...
%                     strjoin(dm_config_reduced.excluded_predictors))
                fde_reduced = obj.get_fraction_deviance_explained(dm_config_reduced);
            end
            
            % the tuning is expressed as a percentage
            tuning = 100*(fde_full - fde_reduced);
        end
        
        function tuning = get_tuning_to_each_predictor(obj, dm_config)
            %GET_TUNING_TO_EACH_PREDICTOR return the tuning of this neuron
            %to all the active predictors, taken individually, and to the
            %special predictor groups, for the given DM config.
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            predictor_names = dm_config.get_active_predictors();
            n_predictors = length(predictor_names);
            
            tuning = zeros(1, n_predictors);
            column_names = cell(1, n_predictors);
            
            % collect tuning indices to simple predictors
            for p_id=1:n_predictors
                predictor_name = predictor_names(p_id);
                tuning(p_id) = obj.get_tuning_index(predictor_name);
                % adjust the names of the predictor to make valid matlab
                % table column names.
                column_names{p_id} = obj.predname2varname(predictor_name);
            end
            if dm_config.use_special_predictor_groups
                % collect tuning indices to special predictor sets
                set_names = dm_config.special_predictor_sets.keys;
                for s_id=1:length(dm_config.special_predictor_sets)
                    tuning_ind_position = n_predictors+s_id;
                    set_name = set_names{s_id};
                    predictor_set = dm_config.get_special_predictor_set(set_name);
                    tuning(tuning_ind_position) = obj.get_tuning_index(predictor_set);
                    column_names{tuning_ind_position} = set_name;
                end
            end
            tuning = array2table(tuning, 'VariableNames', column_names);
        end
        
        function purity = get_tuning_sparseness(obj, dm_config, use_groups)
            %GET_TUNING_SPARSENESS get sparseness, or purity, of tuning of
            %this neuron.
            %
            % If |use_groups| is false (default), the metric is computed
            % only over the simple predictors. If it is true, the metric is
            % computed over the special predictor groups.
            %
            % This measure is 1 when the neuron is tuned to only one
            % predictor, and 0 when it is tuned equally to all predictors
            % in the model. This is defined by analogy to a a very common
            % measure of sparseness, popularized in the neural coding
            % literature by Vinje and Gallant 2000. To give a geometrical
            % intuition, if we consider the k-dimensional vector of tuning
            % to the k predictor in the model, this metric is related
            % (modulo normalization) to the square of the cosine of the
            % angle between this tuning vector and the unit vector that
            % points in the [1,1,...,1] direction (i.e., the direction of
            % "homogeneous" tuning to all predictors).
            if nargin < 2 || isempty(dm_config)
                dm_config = obj.session.model.design_matrix_config;
            end
            if nargin < 3
                use_groups = false;
            end
            tuning = obj.get_tuning_to_each_predictor(dm_config);
            tuning_dimensions = xor(use_groups,...
                ~ismember(tuning.Properties.VariableNames, dm_config.special_predictor_sets.keys));
            tuning = tuning{1,tuning_dimensions};
            tuning(tuning<0) = 0;
            purity = vg_sparseness(tuning);
        end
        
        function [fde_full, fde_reduced] = get_full_and_reduced_fraction_deviance_explained(obj, dm_config)
            %GET_FULL_AND_REDUCED_FRACTION_DEVIANCE_EXPLAINED return the
            %FDE for this neuron under the full model and under all reduced
            %models.
            
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            predictor_names = dm_config.get_active_predictors();
            n_predictors = length(predictor_names);
            
            fde_reduced = zeros(1, n_predictors);
            
            % compute fraction of deviance explained for the full model
            fde_full = obj.get_fraction_deviance_explained(dm_config);
            
            % compute FDEs for reduced models
            for p_id=1:n_predictors
                predictor_name = predictor_names(p_id);
                
                if obj.is_trivial_reduced_model(predictor_name, dm_config)
                    % if the fitted coefficients for this predictor are all
                    % zero, and if |skip_zero_kernel_fde| is true, don't
                    % bother fitting the reduced model and just assumed
                    % that the reduced FDE will be the same as the full FDE
                    this_fde = fde_full;
                else
                    % if this predictor has at least one nonzero
                    % coefficient, build the reduced model and compute its
                    % FDE
                    this_fde = obj.get_fraction_deviance_explained(dm_config.get_reduced_variant(predictor_name));
                end
                    
                fde_reduced(p_id) = this_fde;
            end
            
            % compute FDE for special predictor sets
            if dm_config.use_special_predictor_groups
                set_names = dm_config.special_predictor_sets.keys;
                for s_id=1:length(dm_config.special_predictor_sets)
                    set_name = set_names{s_id};
                    predictor_set = dm_config.get_special_predictor_set(set_name);
                    if obj.is_trivial_reduced_model(predictor_set, dm_config)
                        this_fde = fde_full;
                    else
                        this_fde = obj.get_fraction_deviance_explained(dm_config.get_reduced_variant(predictor_set));
                    end
                    fde_reduced(n_predictors+s_id) = this_fde;
                end
            end
        end
        
        function clear_fits(obj)
            %CLEAR_FITS clear any stored model fit for this neuron
            obj.fit = containers.Map;
        end
        
        function [peak_locs, peak_vals] = get_kernel_peaks(obj, dm_config)
            %GET_KERNELS_PEAKS Return peak location for all kernels for this neuron.
            
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            [kernels, lags] = obj.get_kernels(dm_config);
            
            n_predictors = length(kernels.keys);
            peak_locs = containers.Map;
            peak_vals = containers.Map;

            for i=1:n_predictors
               predictor_name = dm_config.config(i,1).Row{1,1};
               this_kernel = kernels(predictor_name);
               these_lags = lags(predictor_name);
               [peak_val, peak_loc] = max(abs(this_kernel));
               if peak_val == 0
                   peak_locs(predictor_name) = NaN;
                   peak_vals(predictor_name) = NaN;
               else
                   peak_locs(predictor_name) = these_lags(peak_loc)/1000;
                   peak_vals(predictor_name) = peak_val;
               end
            end
        end
        
        function [kernels, lags] = get_kernels(obj, dm_config)
            %GET_KERNELS Return fitted kernels for this neuron.
            
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            n_predictors = height(dm_config.config);
            kernels = containers.Map;
            lags = containers.Map;
            
            for i=1:n_predictors
                predictor_name = dm_config.config(i,1).Row{1,1};
                [kernels(predictor_name), lags(predictor_name)] = obj.get_kernel(predictor_name, dm_config);
            end
        end
        
        
        %% ----PLOTTING METHODS----
        
        function fh = plot_kernels(obj, dm_config)
            
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            [kernels, lags] = obj.get_kernels(dm_config);
            fh = figure();
            
            predictor_names = kernels.keys;
            n_predictors = length(predictor_names);
            
            side = ceil(sqrt(n_predictors));
            
            kernel_scales = [0, 0];
            kernel_scales_internal = [0, 0];
            kernel_scales_reward = [0, 0];
            axh = zeros(1,n_predictors);
            
            for i=1:n_predictors
                predictor_name = predictor_names{1,i};
                axh(i) = subplot(side, side, i);
                hold on
                area(lags(predictor_name), kernels(predictor_name));
                title(strrep(predictor_name, '_', ' '))
                yl = ylim;
                
                if ismember(predictor_name, obj.session.model.design_matrix_config.special_predictor_sets('internal'))
                    kernel_scales_internal(1) = min(kernel_scales_internal(1), min(kernels(predictor_name)));
                    kernel_scales_internal(2) = max(kernel_scales_internal(2), max(kernels(predictor_name)));
                elseif strcmp(predictor_name, "reward_rate")
                    kernel_scales_reward(1) = min(kernel_scales_reward(1), min(kernels(predictor_name)));
                    kernel_scales_reward(2) = max(kernel_scales_reward(2), max(kernels(predictor_name)));
                else
                    kernel_scales(1) = min(kernel_scales(1), min(kernels(predictor_name)));
                    kernel_scales(2) = max(kernel_scales(2), max(kernels(predictor_name)));
                end
                plot([0,0], yl, 'LineStyle', ':')
                xlabel('Time (ms)')
            end
            
            if all(kernel_scales==0)
                % set y axis limits to [-1,1] if no kernel is different
                % from zero
                kernel_scales = [-1, 1];
            end
            if all(kernel_scales_internal==0)
                % set y axis limits to [-1,1] if no kernel is different
                % from zero
                kernel_scales_internal = [-1, 1];
            end
            if all(kernel_scales_reward==0)
                % set y axis limits to [-1,1] if no kernel is different
                % from zero
                kernel_scales_reward = [-1, 1];
            end
            
            for i=1:n_predictors
                if ismember(predictor_names{1,i}, obj.session.model.design_matrix_config.special_predictor_sets('internal'))
                    set(axh(i), 'YLim', 1.2*kernel_scales_internal);
                elseif strcmp(predictor_names{1,i}, "reward_rate")
                    set(axh(i), 'YLim', 1.2*kernel_scales_reward);
                else
                    set(axh(i), 'YLim', 1.2*kernel_scales);
                end
            end
        end
        
        
        function fh = plot_prediction(obj, dm_config, ca)
            
            if nargin < 3
                fh = figure('Position', [0,0,1024,480]);
            else
                axes(ca);
                fh = gcf;
            end
                
            
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            yhat = obj.get_prediction(dm_config);
            
            hold on
            
            plot(obj.session.cam_ts/1000, obj.trace);
            
            prediction_timesteps = obj.session.cam_ts(obj.get_timestep_id_range());
            plot(prediction_timesteps/1000, yhat);
            xlim([0,max(obj.session.cam_ts/1000)]);
            legend('data', 'model prediction')
            xlabel('Time (s)')
            ylabel('Fluorescence')
            
%             yl = ylim();  
%             for e_id=1:height(obj.session.event_data)
%                 t = obj.session.event_data{e_id,'time'};
%                 plot([t,t], yl, 'LineStyle', ':');
%             end
%             ylim(yl)
                
        end
        
        function fh = plot_deviance_summary(obj, dm_config, ca)
            %PLOT_DEVIANCE_SUMMARY visualize FDE for full model and for all
            %"reduced" models obtained by removing one predictor from the
            %full one.
            
            if nargin < 3
                fh = figure();
                add_predictor_names = true;
            else
                axes(ca);
                fh = gcf;
                add_predictor_names = false;
            end
            
            if nargin<2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            [fde_full, fde_reduced] = obj.get_full_and_reduced_fraction_deviance_explained(dm_config);
            
            predictor_names = [dm_config.get_active_predictors(), string(dm_config.special_predictor_sets.keys)];
            n_predictors = length(predictor_names);
            
            hold on
            bar(0, 100*fde_full, 'k');
            bar(1:n_predictors, 100*fde_reduced);
            ylabel("FDE (%)")
            
            if add_predictor_names
                xticks(0:n_predictors);
                xticklabels(["Full", arrayfun(@(s) obj.predname2printname(s), predictor_names)]);
                xtickangle(45);
            end
            
        end
        
        
        function fh = plot_tuning_summary(obj, dm_config, ca)
            %PLOT_TUNING_SUMMARY create spiderplot showing tuning index for
            %each of the predictors in the model.
            %
            %   The length of each arm of the spider is given by the
            %   difference in FDE between the full model and a reduced
            %   model obtained by removing a predictor from the full one.
           
            if nargin < 3
                fh = figure();
                ca = gca();
            else
                axes(ca);
                fh = gcf;
            end
            
            if nargin<2 || isempty(dm_config)
                dm_config = obj.session.model.design_matrix_config;
            end
            
            tuning = obj.get_tuning_to_each_predictor(dm_config);
            predictor_names = tuning.Properties.VariableNames;
            tuning = table2array(tuning)';
            
            % if the autoregressive predictor is present we ignore it, as
            % it would otherwise tend to mess up the plot by being much
            % more influential than the others
            autoregressive_idx = find(strcmp("autoregressive", predictor_names), 1, 'first');
            if ~isempty(autoregressive_idx)
                predictor_names(autoregressive_idx) = [];
                tuning(autoregressive_idx) = [];
            end
            
            spider(...
                tuning,...
                arrayfun(@(s) obj.predname2printname(obj.varname2predname(s)), predictor_names),...
                ca);
        end
        
    end
    
    %% ----UTILITY FUNCTIONS FOR PREDICTOR NAME CONVERSION----
    methods (Static)
        
        function varname = predname2varname(predname)
            %PREDNAME2VARNAME convert predictor name to format that can be
            %used as a column name in a matlab table (i.e. a legal matlab
            %variable name).
            varname = char(strrep(strrep(predname, '-', '_M_'),':','_X_'));
        end
        
        function predname = varname2predname(varname)
            %VARNAME2COLNAME convert predictor name from 'table column
            %name' format to standard format.
            predname = char(strrep(strrep(varname, '_M_', '-'),'_X_',':'));
        end
        
        function printname = predname2printname(predname)
            %PREDNAME2PRINTNAME convert predictor name from standard format
            %to a pretty-printable string.
            printname = strrep(predname,"_"," ");
            if length(char(printname))>19
                printname = char(printname);
                printname = string(printname(1:19));
            end
        end
        
    end
    
    %% ----PRIVATE INTERFACE----
    % These are methods that are only supposed to be used by other
    % components of the library.
    methods (Access=private)
        
        function design_matrix = get_design_matrix(obj, dm_config)
            %GET_DESIGN_MATRIX compute or load design matrix
            %
            %   This loads the design matrix that should be used to fit the
            %   encoding model specified by dm_config for the current
            %   neuron. This method is neuron-specific because in principle
            %   there could be predictors that differ across neurons: for
            %   instance, an autoregressive predictor.
            
            design_matrix = obj.session.get_design_matrix(dm_config);
            
            % if present, replace placeholder column with time-shifted
            % activity
            if ismember("autoregressive", dm_config.config.Properties.RowNames) && ~ismember("autoregressive", dm_config.excluded_predictors)
                autoregressive_columns = dm_config.column_indices('autoregressive');
                for shift=1:dm_config.config{'autoregressive', 'n_knots'}
                    col = autoregressive_columns(shift);
                    design_matrix(1+shift:end,col) = obj.trace(1:end-shift);
                end
                
                
            end
        end
        
        function fit = get_fit(obj, dm_config)
            %GET_FIT compute or load encoding model
            
            fit_name = dm_config.get_model_name();
          
            if ~isKey(obj.fit, fit_name) || obj.session.model.is_new_dm_config(dm_config)

                design_matrix = obj.get_design_matrix(dm_config);                
                timestep_range = obj.get_timestep_id_range();
                fold_id = obj.session.get_folds();
                
                x = design_matrix(timestep_range,:);
                y = obj.trace(timestep_range);
                
                if startsWith(obj.session.model.glm_fit_config.method, 'glmnet')
                    opts = glmnetSet;
                    opts.alpha = obj.session.model.glm_fit_config.alpha;
                    opts.intr = obj.session.model.glm_fit_config.intr;
                    opts.thresh = obj.session.model.glm_fit_config.thresh;
                    opts.nlambda = obj.session.model.glm_fit_config.nlambda;
                    switch obj.session.model.glm_fit_config.method
                        case 'glmnet_matlab'
                            fitting_function = @cvglmnet;
                            parallel = false; % parallelization over CV folds is only supported through R
                        case 'glmnet_R'
                            fitting_function = @cvglmnetR;
                            parallel = obj.session.model.glm_fit_config.parallel;
                    end
                    
                    obj.fit(fit_name) = fitting_function(...
                        x,...
                        y,...
                        'gaussian',...
                        opts,...
                        'deviance',...
                        [],...
                        fold_id(timestep_range),...
                        parallel);
                    
                    
                elseif strcmp(obj.session.model.glm_fit_config.method, 'SGL')
                    opts = SGLSet;
                    opts.alpha = obj.session.model.glm_fit_config.alpha;
                    opts.thresh = obj.session.model.glm_fit_config.thresh;
                    opts.nlambda = obj.session.model.glm_fit_config.nlambda;                    
                    
                    obj.fit(fit_name) = cvSGL(...
                        x,...
                        y,...
                        dm_config.get_group_indices(),...
                        opts,...
                        fold_id(timestep_range));
                    
                elseif strcmp(obj.session.model.glm_fit_config.method, 'gglasso')
                    opts = cvgglassoSet;
                    opts.nlambda = obj.session.model.glm_fit_config.nlambda;                    
                    
                    obj.fit(fit_name) = cvgglasso(...
                        x,...
                        y,...
                        dm_config.get_group_indices(),...
                        opts,...
                        fold_id(timestep_range));                        
                end
            end
            fit = obj.fit(fit_name);
        end
        
        function lambda_index = select_lambda(obj, loss_sequence, loss_error_sequence)
            %SELECT_LAMBDA convenience function to find the appropriate
            %lambda index along a regularization path, similarly to what is
            %done automatically by glmnet, but re-implemented here to be
            %usable with other inference packages also.
            [loss_min, loss_min_index] = min(loss_sequence);
            criterion = obj.session.model.glm_fit_config.lambda_selection_criterion;
            switch criterion
                case 'lambda_min'
                    lambda_index = loss_min_index;
                case 'lambda_1se'
                    lambda_index = find(loss_sequence -(loss_min+loss_error_sequence(loss_min_index)) <= 0, 1, 'first');
            end
        end
        
        function coefs = get_coefs(obj, dm_config)
            %GET_COEFS return fitted coefficients for this neuron.
            %
            %   The coefficients are returned as a column vector.
            if nargin < 2
                dm_config = obj.session.model.design_matrix_config;
            end
                
            if obj.excluded
                coefs = NaN(dm_config.column_counter,1);
            else
                    
                this_fit = obj.get_fit(dm_config);
                
                if startsWith(obj.session.model.glm_fit_config.method, 'glmnet')
                    coefs = cvglmnetCoef(this_fit, obj.session.model.glm_fit_config.lambda_selection_criterion);
                elseif strcmp(obj.session.model.glm_fit_config.method, 'SGL')
                    error('This feature is not implemented yet for SGL.')
                elseif strcmp(obj.session.model.glm_fit_config.method, 'gglasso')
                    % figure out the position of the appropriate lambda in the
                    % sequence, given the criterion
                    lambda_id = obj.select_lambda(this_fit.cvm, this_fit.cvsd);
                    % get raw coefficients
                    coefs = [this_fit.fit.b0(lambda_id); this_fit.fit.beta(:,lambda_id)];
                    if this_fit.standardize
                        % if necessary, invert standardization to put them back
                        % onto the original data scale.
                        coefs(2:end) = coefs(2:end)./this_fit.scaling_scale;
                        coefs(1) = coefs(1) - coefs(2:end)'*this_fit.scaling_center;
                    end
                end
            end
        end
        
        function predictor_coefs = get_predictor_coefs(obj, dm_config, predictor_name)
            %GET_PREDICTOR_COEFS get fitted coefficients associated to the
            %given predictor
            
            if nargin<2
                dm_config = obj.session.model.design_matrix_config;
            end
            
            % note that the coefficient array returned by glmnetCoef always
            % has one more element than the number of columns of the design
            % matrix. This is meant to store the intercept coefficient, but
            % is present even if the model was fit without an intercept
            % term! In this latter case, the coefficient will always be
            % zero.
            column_shift = 1;
            
            column_indices = dm_config.column_indices(char(predictor_name));
            all_coefs = obj.get_coefs(dm_config);
            
            predictor_coefs = all_coefs(column_shift+column_indices);
        end
    end
    
    methods (Access=?experiment)
        function [kernel, lags] = get_kernel(obj, predictor_name, dm_config)
            if nargin < 3
                dm_config = obj.session.model.design_matrix_config;
            end
            predictor_coefs = obj.get_predictor_coefs(dm_config, predictor_name);
            [kernel, lags] = obj.session.model.get_kernel(dm_config, predictor_name, predictor_coefs);
        end
    end
    
    methods (Access={?session, ?population, ?experiment})
        
        function timestep_range = get_timestep_id_range(obj)
            %GET_TIMESTEP_ID_RANGE get the indices of the timesteps contained
            %in the active trials for this neuron.
            timestep_range = obj.session.get_timestep_id_range(obj.active_trials);
        end
        
        function test = is_trivial_reduced_model(obj, predictors, dm_config)
            %IS_TRIVIAL_REDUCED_MODEL return whether the reduced model
            %obtained by removing the given group of predictors is not (in
            %principle) different from the full model because the fitted
            %parameters for the given predictors in the full model are all
            %zero.
            %
            %   Note that the behavior of this method is affected by the
            %   skip_zero_kernel_fde property of glmFitConfiguration, which
            %   will guarantee that all reduced models will be considered
            %   'nontrivial'.
            coefs = [];
            test = obj.excluded; % models are always considered trivial if they refer to an excluded neuron
            if ~test
                for p_id=1:length(predictors)
                    predictor = predictors(p_id);
                    coefs = [coefs, obj.get_predictor_coefs(dm_config, predictor)'];
                end
                test = ~any(coefs) && obj.session.model.glm_fit_config.skip_zero_kernel_fde;
            end
        end
             
    end
end

