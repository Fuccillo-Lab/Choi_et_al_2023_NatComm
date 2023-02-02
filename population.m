classdef population < handle
    
    properties (Constant)
        nboot = 1e5; % number of bootstrap samples for estimation of CI for tuning indices
        kernel_nboot = 1e4; % number of bootstrap samples for estimation of CI and significance tests for population kernels
        ci_alpha = 0.16 % alpha value for confidence interval
    end
    
    properties
        neurons = neuron.empty()
        name = ''
        tuning
        tuning_ci
        single_neuron_kernels = containers.Map;
        single_neuron_lags = containers.Map;
        experiment
    end
    
    methods
        
        function obj = population(name, experiment)
            if nargin > 0
                obj.name = name;
            end
            if nargin > 1
                obj.experiment = experiment;
            end
        end
        
        function add_neurons(obj, neurons)
            obj.neurons = [obj. neurons, neurons];
        end
        
        function sparseness = get_tuning_sparseness(obj, dm_config, use_groups, tuning_threshold)
            %GET_TUNING_SPARSENESS return tuning sparseness for all
            %non-excluded neurons in this population.
            if nargin < 2
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            if nargin < 3
                use_groups = false;
            end
            if nargin < 4
                tuning_threshold = 0;
            end
            
            neurons_in_use = obj.get_neurons_in_use();
            
            % if requested, only keep those neurons that meet the given
            % tuning threshold
            if tuning_threshold > 0
                neurons_in_use_thresholded = neuron.empty();
                for n=1:length(neurons_in_use)
                    if neurons_in_use(n).get_fraction_deviance_explained() >= tuning_threshold
                        neurons_in_use_thresholded = [neurons_in_use_thresholded neurons_in_use(n)];
                    end
                end
                neurons_in_use = neurons_in_use_thresholded;
            end
            
            % tally sparnesess distribution across population
            sparseness = zeros(1,length(neurons_in_use));
            for n=1:length(neurons_in_use)
                this_neuron = neurons_in_use(n);
                sparseness(n) = this_neuron.get_tuning_sparseness(dm_config, use_groups);
            end
            
        end
        
        function tuning = get_single_neuron_tuning(obj, dm_config)
            %GET_TUNING Compute tuning index to all individual predictors,
            %for all neurons.
            %
            %   |tuning| is an n_neurons x n_predictors table of tuning
            %   indices.
            
            if nargin < 2
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            
            tuning = table();
            
            for n=1:length(obj.neurons)
                tuning = [tuning; obj.neurons(n).get_tuning_to_each_predictor(dm_config)];
            end
        end
        
        function [tuning, tuning_ci] = get_tuning_to_each_predictor(obj, dm_config)
            %GET_TUNING_TO_EACH_PREDICTOR return the tuning of this
            %population to all the active predictors, taken individually,
            %and to the special predictor groups, for the given DM config.
            if nargin < 2
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            
            if isempty(obj.tuning)
                predictor_names = ["total", dm_config.get_active_predictors()];
                n_predictors = length(predictor_names);
                
                tuning = zeros(1, n_predictors);
                tuning_ci = zeros(2, n_predictors);
                column_names = cell(1, n_predictors);
                
                % collect tuning indices to simple predictors
                for p_id=1:n_predictors
                    predictor_name = predictor_names(p_id);
                    [tuning(p_id), tuning_ci(:,p_id)] = obj.get_tuning_index(predictor_name, dm_config);
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
                        [tuning(tuning_ind_position), tuning_ci(:,tuning_ind_position)] = obj.get_tuning_index(predictor_set, dm_config);
                        column_names{tuning_ind_position} = set_name;
                    end
                end
                obj.tuning = array2table(tuning, 'VariableNames', column_names);
                obj.tuning_ci = array2table(tuning_ci, 'VariableNames', column_names, 'RowNames', {'CI lower', 'CI upper'});
            end
            tuning = obj.tuning;
            tuning_ci = obj.tuning_ci;
        end
        
        function neuron_subset = get_neurons_with_tuning_criterion(obj, total_tuning_threshold, predictor_for_threshold, predictor_tuning_threshold, dm_config)
            
            if nargin < 2 || isempty(total_tuning_threshold)
                total_tuning_threshold = -Inf;
            end
            if nargin < 3
                predictor_for_threshold = '';
                predictor_tuning_threshold = -Inf;
            end
            if nargin < 5
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            
            % convert predictor name to tuning-table column name
            predictor_column_name = obj.neurons(1).predname2varname(predictor_for_threshold);
            
            
            fde = zeros(length(obj.neurons), 1);
            for n=1:length(obj.neurons)
                fde(n) = 100*obj.neurons(n).get_fraction_deviance_explained(dm_config);

            end
            
            predictor_tuning = zeros(length(obj.neurons), 1);
            if ~isempty(predictor_column_name)
                % note that we have to pass through the
                % get_tuning_to_each_predictor function in order to be able
                % to threshold on group predictors as well.
                all_tuning = obj.get_single_neuron_tuning(dm_config);
                predictor_tuning = all_tuning{:,predictor_column_name};                
            end
            neuron_subset = find((fde > total_tuning_threshold) & (predictor_tuning > predictor_tuning_threshold));
            
            
        end
        
        
        function clear_tuning(obj)
            obj.tuning = [];
            obj.tuning_ci = [];
        end
        
        function neurons_in_use = get_neurons_in_use(obj, neuron_subset)
            %NEURONS_IN_USE select neurons that are not excluded from the
            %analysis
            if nargin < 2
                neuron_subset = [];
            end
            if isempty(neuron_subset)
                candidates = 1:length(obj.neurons);
            else
                candidates = neuron_subset;
            end
            neurons_in_use = neuron.empty();
            for n_id=1:length(candidates)
                this_neuron = obj.neurons(candidates(n_id));
                if ~this_neuron.excluded
                    neurons_in_use = [neurons_in_use, this_neuron];
                end
            end
        end        

        
        function [peak_locs, peak_vals] = get_kernel_peak_distribution(obj, dm_config, neuron_subset)
            
            if nargin < 2
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            if nargin < 3
                neuron_subset = [];
            end
           
            [kernels, lags] = obj.get_all_single_neuron_kernels(dm_config, neuron_subset);
            
            n_predictors = length(kernels.keys);
            peak_locs = containers.Map;
            peak_vals = containers.Map;

            for i=1:n_predictors
               predictor_name = dm_config.config(i,1).Row{1,1};
               if strcmp(predictor_name, 'choice_1:outcome_1')
                   1;
               end
               these_kernels = kernels(predictor_name);
               these_lags = lags(predictor_name);
               [peak_val, peak_loc] = max(abs(these_kernels), [], 1);
               no_max = peak_val==0;
               these_peak_locs = these_lags(peak_loc)/1000;
               these_peak_vals = peak_val;
               these_peak_locs(no_max) = NaN;
               these_peak_vals(no_max) = NaN;
               peak_locs(predictor_name) = these_peak_locs;
               peak_vals(predictor_name) = these_peak_vals;
            end
            
        end
        
        function [kernels_rms, kernels_ci, lags] = get_population_kernels(obj, dm_config, parallel_workers, neuron_subset, get_confidence_intervals)
            %GET_POPULATION_KERNEL compute all mean-displacement "kernels"
            %for this population.
            %
            %   A mean displacement kernel for a predictor is the
            %   root-mean-square of the kernels for that predictor over all
            %   the neurons making up the population.
            %
            %   If |parallel_workers| is empty, this will be run serially.
            %   If it is not empty and a parallel pool already exists, the
            %   existing pool will be used. If a pool does not exist, a
            %   pool of size |parallel_workers| will be created. If
            %   |parallel_workers| is 0, the default number of workers will
            %   be used.
            %
            %   The optional argument |neuron_subset| can be a list of
            %   neuron indices indicating the subset of the neurons in the
            %   population that should be taken into account when computing
            %   the kernel.
            %
            %   If |get_confidence_intervals| is true (default), bootstrap
            %   confidence intervals will be computed. Otherwise, the
            %   |kernels_ci| return argument will be empty.
            
            if nargin < 2
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            if nargin < 3
                parallel_workers = [];
            end
            if nargin < 4
                neuron_subset = [];
            end
            if nargin < 5
                get_confidence_intervals = true;
            end
            
            n_predictors = height(dm_config.config);
            
            lags = containers.Map;
            kernels_ci = containers.Map;
            kernels_rms = containers.Map;
            
            for i=1:n_predictors
                predictor_name = dm_config.config(i,1).Row{1,1};
                [kernels_rms(predictor_name), kernels_ci(predictor_name), lags(predictor_name)] = get_individual_population_kernel(obj,...
                    predictor_name, dm_config, parallel_workers, neuron_subset, get_confidence_intervals);
            end
            
        end
        
        function [kernel_rms, kernel_ci, lags] = get_individual_population_kernel(obj, predictor_name, dm_config, parallel_workers, neuron_subset, get_confidence_intervals)
            % GET_INDIVIDUAL_POPULATION_KERNEL get RMS population kernel
            % for an individual predictor.
            %
            %   For details on how to use this, see GET_POPULATION_KERNELS.
            %   The main differences are that here you need to specify
            %   which predictor you want the kernel for with
            %   |predictor_name|, and that the outputs of the function are
            %   simple arrays, and not Maps.
            if nargin < 3
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            if nargin < 4
                parallel_workers = [];
            end
            if ~isempty(parallel_workers)
                if isempty(gcp('nocreate'))
                    if parallel_workers==0
                        parpool('local');
                    else
                        parpool('local', parallel_workers);
                    end
                end
            end
            if nargin < 5
                neuron_subset = [];
            end
            if nargin < 6
                get_confidence_intervals = true;
            end
            
            [kernels, lags] = obj.get_all_single_neuron_kernels(dm_config, neuron_subset);
            
            lags = lags(predictor_name);
            ci_opts = statset();
            ci_opts.UseParallel = true;
            if get_confidence_intervals
                kernel_ci = bootci(obj.kernel_nboot, {@rms, kernels(predictor_name)', 1},...
                    'alpha',obj.ci_alpha,'Options',ci_opts)';
            else
                kernel_ci = [];
            end
            kernel_rms = rms(kernels(predictor_name),2);
            
        end
        
        
        function [tuning, tuning_ci] = get_tuning_index(obj, predictors, dm_config, equalize_neuron_weight)
            %GET_TUNING_INDEX return (pseudo)population tuning index and CI
            %
            %   Note that this is different from the population tuning
            %   index defined at the level of individual sessions. Most
            %   notably, the index defined here is a simple measure of
            %   change in the total variance for a given pseudopopulation
            %   when one (or more) model predictor(s) is (are) removed.
            %
            %   |predictors| can be any predictor or predictor set. If the
            %   special predictor "total" is specified, then the full model
            %   will be compared with the null (intercept model) to yield
            %   the "total" tuning of the population.
            %
            %   If |equalize_neuron_weight| is true, the tuning index is
            %   simply the mean of the tuning indices of the individual
            %   neurons in the population. If it is false, is it instead
            %   the reduction in fraction of variance explained in the
            %   population activity when one or more predictors are
            %   removed. The two definitions would coincide if all neurons
            %   had the same variance, so the first can be seen as applying
            %   the second after z-scoring all components of the population
            %   activity.
            
            if nargin < 4
                equalize_neuron_weight = true;
            end
            if nargin < 3
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            
            % make sure predictors are specified as strings and not chars
            if ischar(predictors) || iscell(predictors)
                predictors = string(predictors);
            end
            
            neurons_in_use = obj.get_neurons_in_use();
            
            % compute the variance of the data for each neuron as well as
            % the RSS under both the full and reduced model
            tot_var = zeros(1, length(neurons_in_use));
            full_var = zeros(1, length(neurons_in_use));
            reduced_var = zeros(1, length(neurons_in_use));
            
            for n=1:length(neurons_in_use)
                this_neuron = neurons_in_use(n);
                full_dm = this_neuron.get_full_dm_config(predictors, dm_config);
                trace = this_neuron.trace(this_neuron.get_timestep_id_range());
                full_pred = this_neuron.get_prediction(full_dm);
                
                if (length(predictors)==1 && strcmp("total", predictors)) || this_neuron.is_trivial_reduced_model(predictors, full_dm)
                    reduced_pred = full_pred;
                else
                    reduced_pred = this_neuron.get_prediction(full_dm.get_reduced_variant(predictors));
                end
                
                full_residuals = trace - full_pred;
                reduced_residuals = trace - reduced_pred;
                
                tot_var(n) = var(trace);
                full_var(n) = var(full_residuals);
                reduced_var(n) = var(reduced_residuals);
            end
            
            if strcmp("total", predictors)
                reduced_var = tot_var;
            end
            
            if equalize_neuron_weight
                tuning_fn = @(reduced, full, tot) 100*mean((reduced-full)./tot);
            else
                tuning_fn = @(reduced, full, tot) 100*sum(reduced-full)/sum(tot);
            end
            tuning = tuning_fn(reduced_var, full_var, tot_var);
            % bootstrap confidence interval
            tuning_ci =  bootci(obj.nboot,...
                {tuning_fn, reduced_var, full_var, tot_var},...
                'alpha', obj.ci_alpha);
        end
        
        %% ----PLOTTING METHODS----
        function plot_choice_outcome_mixed_tuning(obj)
            neurons_in_use = obj.get_neurons_in_use();
            n_neurons = length(neurons_in_use);
            choice = NaN(n_neurons,1);
            cXoutcome = NaN(n_neurons,1);
            for n=1:n_neurons
                neuron = neurons_in_use(n);
                choice(n) = neuron.get_tuning_index({'choice_1', 'choice_-1'});
                cXoutcome(n) = neuron.get_tuning_index({'choice_1:outcome_1', 'choice_1:outcome_-1'});
            end
            choice(choice<0) = 0;
            cXoutcome(cXoutcome<0) = 0;
            scatter(choice, cXoutcome, 'MarkerEdgeColor', obj.experiment.colors_lines(obj.name));
        end
        
        function fh = plot_population_kernels(obj, fh, dm_config, parallel_workers, neuron_subset)
            %PLOT_POPULATION_KERNELS summary plot of the mean-displacement
            %kernels for this population.
            %
            %   Note that this code is lifted almost identically from the
            %   corresponding method of |neuron|.
            
            if nargin < 3
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            if nargin < 4
                parallel_workers = [];
            end
            if nargin < 5
                neuron_subset = [];
            end
            
            [kernels, kernels_ci, lags] = obj.get_population_kernels(dm_config, parallel_workers, neuron_subset);
            
            predictor_names = kernels.keys;
            n_predictors = length(predictor_names);
            side = ceil(sqrt(n_predictors));
            kernel_scales = [0, 0];
            kernel_scales_internal = [0, 0];
            kernel_scales_reward = [0, 0];
            
            if nargin < 2 || isempty(fh)
                fh = figure('Name', sprintf("Mean-displacement population kernels - %s", obj.name));
            else
                figure(fh);
                % if we are plotting on an existing figure, take into
                % account existing ylims to determine final ylims
                subplot(side,side,...
                    find(~ismember(predictor_names, dm_config.special_predictor_sets('internal')), 1));
                kernel_scales = ylim./1.2;
                if any(ismember(predictor_names, dm_config.special_predictor_sets('internal')))
                    subplot(side,side,...
                        find(ismember(predictor_names, dm_config.special_predictor_sets('internal')), 1));
                    kernel_scales_internal = ylim./1.2;
                end
                if any(strcmp("reward_rate", predictor_names))
                    subplot(side,side,...
                        find(strcmp("reward_rate", predictor_names), 1));
                    kernel_scales_reward = ylim./1.2;
                end

            end
            
            axh = zeros(1,n_predictors);
            
            for i=1:n_predictors
                predictor_name = predictor_names{1,i};
                axh(i) = subplot(side, side, i);
                hold on
                ci = kernels_ci(predictor_name);
                if any(ci(:))
                    lh = fill_between(lags(predictor_name), ci(:,1), ci(:,2));
                    set(lh, 'FaceColor',...
                        obj.neurons(1).session.experiment.colors(obj.name),...
                        'EdgeColor', obj.neurons(1).session.experiment.colors_lines(obj.name),...
                        'FaceAlpha', 0.3, 'LineWidth', 0.5)
                end
                plot(lags(predictor_name), kernels(predictor_name),...
                    'Color', obj.neurons(1).session.experiment.colors_lines(obj.name),...
                    'LineWidth',2)
%                 area(lags(predictor_name), kernels(predictor_name),...
%                     'FaceColor', obj.neurons(1).session.experiment.colors(obj.name),...
%                     'EdgeColor', obj.neurons(1).session.experiment.colors_lines(obj.name),...
%                     'FaceAlpha', 0.6);
                title(strrep(predictor_name, '_', ' '))
                yl = ylim;
                
                if ismember(predictor_name, dm_config.special_predictor_sets('internal'))
                    kernel_scales_internal(1) = min(kernel_scales_internal(1), min(ci(:,1)));
                    kernel_scales_internal(2) = max(kernel_scales_internal(2), max(ci(:,2)));
                elseif strcmp(predictor_name, "reward_rate")
                    kernel_scales_reward(1) = min(kernel_scales_reward(1), min(ci(:,1)));
                    kernel_scales_reward(2) = max(kernel_scales_reward(2), max(ci(:,2)));
                else
                    kernel_scales(1) = min(kernel_scales(1), min(ci(:,1)));
                    kernel_scales(2) = max(kernel_scales(2), max(ci(:,2)));
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
                if ismember(predictor_names{1,i}, dm_config.special_predictor_sets('internal'))
                    set(axh(i), 'YLim', 1.2*kernel_scales_internal);
                elseif strcmp("reward_rate", predictor_names{1,i})
                    set(axh(i), 'YLim', 1.2*kernel_scales_reward);
                else
                    set(axh(i), 'YLim', 1.2*kernel_scales);
                end
            end
        end
        
        function fh = plot_individual_kernels(obj, predictor_name, neuron_subset, dm_config)
            %PLOT_INDIVIDUAL_KERNELS compare time course of individual
            %neuron kernels with the population kernel.
            
            if nargin < 3
                neuron_subset = [];
            end
            if nargin < 4
                dm_config = obj.neurons(1).session.model.design_matrix_config;
            end
            obj.get_all_single_neuron_kernels(dm_config, []);
            kernels = obj.single_neuron_kernels(predictor_name)';
            lags = obj.single_neuron_lags(predictor_name);
            
            % rectify and normalize kernels for plotting
            if ~isempty(neuron_subset)
                kernels = kernels(neuron_subset, :);
            end
            n_neurons = size(kernels, 1);
            kernels = abs(kernels);
            
            
            % determine sorting order of neurons
            [peak_locs, ~] = obj.get_kernel_peak_distribution(dm_config, neuron_subset);
            peak_locs = peak_locs(predictor_name);
            [peak_locs, peak_locs_sorting] = sort(peak_locs);
            
            kernels = kernels(peak_locs_sorting, :);
            
            % normalize kernels by their peak
            normkernels = kernels ./ max(kernels, [], 2);
            
            fh = figure('Name', sprintf("Individual %s kenels - %s", obj.neurons(1).predname2printname(predictor_name), obj.name),...
                'Position', [400, 400, 1140, 460]);    
            % plot kernels
            subplot(1,3,1);
            imagesc(normkernels, 'XData', [lags(1), lags(end)]/1000);
            xlabel('Lag from event (s)')
            ylabel('Neuron')
            title('Max-normalized kernels (unaligned)')
            
            % plot peak shape
            subplot(1,3,2);
            window_length = size(kernels, 2);
            window_length_s = (lags(end)-lags(1))/1000;
            kernel_res = lags(2)-lags(1);
            
            peak_shape = NaN(n_neurons, 2*window_length-1);
            for n=1:n_neurons
                offset = window_length - round(1000*peak_locs(n)/kernel_res) - 1;
                if isfinite(offset)
                    peak_shape(n,offset:offset+window_length-1) = kernels(n,:);
                end
            end
            
            imagesc(peak_shape, 'XData', [-window_length_s, window_length_s])
            xlabel('Lag from peak (s)')
            ylabel('Neuron')
            title('Max-aligned kernels (unnormalized)')
            
            
            subplot(1,3,3);
            
            hold on
            lagvals = -window_length_s:kernel_res/1000:window_length_s;
            xlabel('Lag from peak (s)')
            ylabel('Max-normalized population kernel')
            
            % compare with population kernel
            popkernel = sqrt(mean(kernels.*2, 1, 'omitnan'));
            [~, I] = max(popkernel);
            popkernel_centered = zeros(1, 2*window_length-1);
            offset = window_length - I - 1;
            popkernel_centered(offset:offset+window_length-1) = popkernel;
            popkernel_centered = popkernel_centered/max(popkernel_centered);
            
            meankernel_centered = sqrt(mean(peak_shape.*2, 1, 'omitnan'));
            meankernel_centered = meankernel_centered/max(meankernel_centered);
            
            plot(lagvals, popkernel_centered, 'Color', obj.experiment.colors_lines(obj.name));
            plot(lagvals, meankernel_centered, 'Color', obj.experiment.colors(obj.name), 'Linestyle', '--');
                        
            title(sprintf('FWHM: pop %.2gs, aligned neurons %.2gs',...
                kernel_res/1000*obj.fwhm(popkernel),...
                kernel_res/1000*obj.fwhm(meankernel_centered)))
            
            legend({'Population kernel', 'Aligned pop kernel'}, 'Location', 'south')
            
            
            
%             % compute temporal sparseness
%             sparseness = vg_sparseness(kernels,2);
%             subplot(2,2,4)
%             hold on
%             scatter(1:size(sparseness),sparseness, 'MarkerFaceColor', obj.experiment.colors(obj.name), 'MarkerEdgeColor', obj.experiment.colors_lines(obj.name))
%             xl = xlim;
%             plot(xl, [vg_sparseness(popkernel), vg_sparseness(popkernel)], 'Color', obj.experiment.colors_lines(obj.name));
%             xlim(xl);
        end
       
    end
    
    methods (Access=public)
        function [kernels, lags] = get_all_single_neuron_kernels(obj, dm_config, neuron_subset)
            %GET_ALL_SINGLE_NEURON_KERNELS extract all kernels for all
            %predictors for all infidividual neurons of this population.
            
            
            n_predictors = height(dm_config.config);
            neurons_in_use = obj.get_neurons_in_use([]);
            n_neurons = length(neurons_in_use);
            n_total_neurons = length(obj.neurons);
            
            % if necessary, store data structure of all kernels for this
            % population
            if isempty(obj.single_neuron_kernels)
                
                kernels = containers.Map;
                lags = containers.Map;
            
                for i=1:n_predictors
                    predictor_name = dm_config.config(i,1).Row{1,1};
                    kernels(predictor_name) = [];
                end

                for n_id=1:n_neurons
                    [kernels_temp, lags_temp] = neurons_in_use(n_id).get_kernels(dm_config);
                    for i=1:n_predictors
                        predictor_name = dm_config.config(i,1).Row{1,1};
                        if n_id==1
                            lags(predictor_name) = lags_temp(predictor_name);
                        end
                        kernels(predictor_name) = [kernels(predictor_name), kernels_temp(predictor_name)];
                    end
                end
                obj.single_neuron_kernels = kernels;
                obj.single_neuron_lags = lags;
            end
            
            % from an existing pre-computed data structure containing
            % kernels, extract the relevant ones
            if ~isempty(neuron_subset)
                kernels = containers.Map;
                lags = containers.Map;
                % merge information about a priori 'excluded' neurons and
                % desired neurons for this function call to work out
                % exactly which ones are the kernels to return. Remember
                % that "neuron_subset" contains indices with respect to the
                % total array of neurons, which includes also "excluded"
                % neurons, but the matrix stored in "kernels" has a number
                % of rows equal to the number of non-excluded neurons.
                selected_neurons = false(1,n_neurons);
                non_excluded_counter = 0;
                for n_absolute_id=1:n_total_neurons
                    if ~obj.neurons(n_absolute_id).excluded
                        non_excluded_counter = non_excluded_counter + 1;
                        if ismember(n_absolute_id, neuron_subset)
                            selected_neurons(non_excluded_counter) = true;
                        end
                    end
                end
                for i=1:n_predictors
                    predictor_name = dm_config.config(i,1).Row{1,1};
                    these_kernels = obj.single_neuron_kernels(predictor_name);
                    kernels(predictor_name) = these_kernels(:,selected_neurons);
                    lags(predictor_name) = obj.single_neuron_lags(predictor_name);
                end
            else
                kernels = obj.single_neuron_kernels;
                lags = obj.single_neuron_lags;
            end
            
        end 
    end
    
    methods (Static, Access=private)
        function y = rms(x, dim)
            y = sqrt(mean(x.^2, dim));
        end
        
        function w = fwhm(y)
            above = y > max(y)/2;
            w = find(above, 1, 'last') - find(above, 1, 'first');
        end
    end
       
    
end