classdef experiment < handle
    %EXPERIMENT Multiple sessions with their analyses
    %   An experiment contains multiple sessions, together with methods to
    %   analyse their data and the analysis results. It also contains
    %   the information needed to build the design matrix for a session,
    %   and to fit the encoding model (GLM).
    
    properties
        sessions
        populations = containers.Map
        model
    end
    
    properties (Constant)
        colors = containers.Map({'aDMS', 'pDMS'},...
            {[85, 95, 160]/255, [232, 156, 66]/255});
        colors_lines = containers.Map({'aDMS', 'pDMS'},...
            {[34, 45, 114]/255, [210, 126, 25]/255});j
        color_kernel_difference_test = [0.9290 0.6940 0.1250];
    end
    
    methods
        %% ----PUBLIC INTERFACE----
        function obj = experiment(sessions, design_matrix_config, glm_fit_config)
            %EXPERIMENT Construct an experiment from a set of sessions
            
            obj.model = encodingModel(design_matrix_config, glm_fit_config);
            obj.model.experiment = obj;

            obj.sessions = sessions;
            for s_id=1:length(obj.sessions)
                obj.sessions(s_id).experiment = obj;
                obj.sessions(s_id).model = obj.model;
            end
            % make sure the loaded sessions do not contain any pre-computed
            % design matrices or model fits. This check is necessary
            % because if they do, we have no way of ensuring consistency of
            % those DMs and fits with the current design matrix
            % configuration.
            obj.clear_dms_and_fits();
            
        end
        
        function set_design_matrix(obj, dm_config)
            %SET_DESIGN_MATRIX store a new DM config and remove any
            %existing DMs and fits. If the new DM config matches the old
            %one, don't do anything.
            if obj.model.is_new_dm_config(dm_config)
                obj.model.design_matrix_config = dm_config.copy();
                obj.clear_dms_and_fits()
            end
        end
        
        function fit_all_models(obj, parallel_workers, dm_config)
            %FIT_ALL_MODELS fit full and reduced models for all neurons.
            %
            %   This is mostly intended here as a convenience function that
            %   allows parallelizing the most computationally expensive
            %   step of the analysis across the neurons.
            %
            %   If |parallel_workers| is empty, this will be run serially.
            %   If it is not empty and a parallel pool already exists, the
            %   existing pool will be used. If a pool does not exist, a
            %   pool of size |parallel_workers| will be created. If
            %   |parallel_workers| is 0, the default number of workers will
            %   be used.
            if nargin < 3
                dm_config = obj.model.design_matrix_config;
            end
            for s_id=1:length(obj.sessions)
                obj.sessions(s_id).fit_all_models(dm_config, parallel_workers);
            end
        end
        
        function tuning = get_tuning(obj, dm_config)
            %GET_TUNING Compute tuning index to all individual predictors,
            %for all neurons in all sessions
            
            if nargin < 2
                dm_config = obj.model.design_matrix_config;
            end
            
            tuning = table();
            
            for s_id=1:length(obj.sessions)
                session = obj.sessions(s_id);
                n_neurons = length(session.neurons);
                s_tuning = session.get_tuning(dm_config);
                s_tuning = addvars(s_tuning,...
                    s_id*ones(n_neurons,1),...
                    (1:n_neurons)',...
                    'Before', 1, 'NewVariableNames', {'Session', 'Neuron'});
                tuning = [tuning; s_tuning];
            end
        end
        
        function tuning = get_population_tuning(obj, dm_config)
            %GET_TUNING Compute population tuning index to all individual
            %predictors, for each session
            
            if nargin < 2
                dm_config = obj.model.design_matrix_config;
            end
            
            tuning = table();
            
            for s_id=1:length(obj.sessions)
                session = obj.sessions(s_id);
                s_tuning = session.get_population_tuning_to_each_predictor(dm_config);
                s_tuning = addvars(s_tuning,...
                    s_id,...
                    {session.pathway},...
                    'Before', 1, 'NewVariableNames', {'Session', 'Pathway'});
                tuning = [tuning; s_tuning];
            end
        end
        
        function populations = get_pathway_populations(obj)
            %GET_PATHWAY_POPULATIONS build populations corresponding to the
            %pathways defined in the sessions' |pathway| attribute.
            if isempty(obj.populations)
                pathways = unique({obj.sessions.pathway});
                obj.populations = containers.Map;
                for p_id=1:length(pathways)
                    pathway = pathways{p_id};
                    obj.populations(pathway) = population(pathway, obj);
                    pop = obj.populations(pathway);
                    for s_id=1:length(obj.sessions)
                        session = obj.sessions(s_id);
                        if strcmp(session.pathway, pathway)
                            pop.add_neurons(session.neurons);
                        end
                    end
                end
            end
            populations = obj.populations;
        end
        
        function clear_populations(obj)
            %CLEAR_POPULATIONS clear from memory any population that may
            %have been already defined.
            %
            %   This is useful if for some reason you want to re-compute
            %   population tuning, as it also clears any precomputed
            %   population tuning.
            obj.populations = population.empty;
        end
        
        function pruned_experiment = export_neuron(obj, session_id, neuron_id)
            %EXPORT_NEURON return experiment object pruned down to only the
            %data that refers to the specified neuron in the specified
            %session.
            
            new_sessions = session.empty();
            for s_id=1:length(obj.sessions)
                new_sessions(s_id) = session();
            end
            new_sessions(session_id) = obj.sessions(session_id).prune(neuron_id);
            pruned_experiment = experiment(new_sessions, obj.model.design_matrix_config, obj.model.glm_fit_config);
        end
        
        function [tests, lags, pvalues] = test_single_pathway_kernel_against_another_experiment(obj,...
                other_exp, pathway_name, predictor_name, dm_config, parallel_workers, alph)
            %TEST_SINGLE_PATHWAY_KERNEL_AGAINST_ANOTHER_EXPERIMENT this can
            %be used to compare statistically a certain kernel of a certain
            %pathway against its analogue in a different experiment, such as
            %when comparing early vs late training sessions.
            %
            %   WARNING: this method is a bit of a hack and assumes that
            %   both experiments are set up in exactly the same way except
            %   for the data.
            if nargin < 5
                dm_config = obj.model.design_matrix_config;
            end
            if nargin < 6
                parallel_workers = [];
            end
            if nargin < 7
                alph = 0.05;
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
            exps = {obj, other_exp};
            kernels_from_exps = {[], []};
            obj.get_pathway_populations;
            
            for e_id=1:2
                exp = exps{e_id};
                pathway = exp.populations(pathway_name);
                
                neurons_in_use = pathway.get_neurons_in_use();
                n_neurons = length(neurons_in_use);
                for n_id=1:n_neurons
                    [kernel_temp, lags] = neurons_in_use(n_id).get_kernel(predictor_name, dm_config);
                    kernels_from_exps{e_id} = [kernels_from_exps{e_id}, kernel_temp];
                end
                
            end
            
            ci_opts = statset();
            ci_opts.UseParallel = true;
            [tests, pvalues] = obj.kernel_diff_bootstraptest(kernels_from_exps{1}',...
                kernels_from_exps{2}', obj.populations(pathway_name).kernel_nboot, alph, ci_opts);
            
        end
        
        function [tests, lags, pvalues] = pathway_kernel_difference_test(obj, dm_config, parallel_workers, alph)
            %PATHWAY_KERNEL_DIFFERENCE_TEST perform bootstrap test for
            %difference of mean-displacement population kernels.
            %
            %   alph: significance level (default: 0.05)
            
            if nargin < 2
                dm_config = obj.model.design_matrix_config;
            end
            if nargin < 3
                parallel_workers = [];
            end
            if nargin < 4
                alph = 0.05;
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
           
            obj.get_pathway_populations;
            pathway_names = {'aDMS', 'pDMS'};
            
            n_predictors = height(dm_config.config);
            
            ci_opts = statset();
            ci_opts.UseParallel = true;
            
            tests = containers.Map;
            pvalues = containers.Map;
            lags = containers.Map;
            kernels = {};
            
            for p_id = 1:length(pathway_names)
                pathway_name = pathway_names{p_id};
                pathway = obj.populations(pathway_name);
                
                this_pathway_kernels = containers.Map;
                for i=1:n_predictors
                    predictor_name = dm_config.config(i,1).Row{1,1};
                    this_pathway_kernels(predictor_name) = [];
                end

                neurons_in_use = pathway.get_neurons_in_use();
                n_neurons = length(neurons_in_use);
                for n_id=1:n_neurons
                    [kernels_temp, lags_temp] = neurons_in_use(n_id).get_kernels(dm_config);
                    for i=1:n_predictors
                        predictor_name = dm_config.config(i,1).Row{1,1};
                        if n_id==1
                            lags(predictor_name) = lags_temp(predictor_name);
                        end
                        this_pathway_kernels(predictor_name) = [this_pathway_kernels(predictor_name), kernels_temp(predictor_name)];
                    end
                end
                
                kernels{p_id} = this_pathway_kernels;
            end

            for i=1:n_predictors
                predictor_name = dm_config.config(i,1).Row{1,1};
                [tests(predictor_name), pvalues(predictor_name)] = obj.kernel_diff_bootstraptest(kernels{1}(predictor_name)',...
                    kernels{2}(predictor_name)', obj.populations(pathway_names{1}).kernel_nboot, alph, ci_opts);
            end
            
        end
                
        %% ----PLOTTING METHODS----
        %
        % Note that many of these methods do the same thing, namely calling
        % a session-specific plotting function on each of the sessions in
        % the experiment, and then renaming the resulting figure. They
        % could be easily refactored into just one generic, more flexible
        % method, but for the moment we keep things as they are for
        % simplicity's sake.
        %
        % Also, note that as they are summary methods for the whole
        % experiment, it is probably a good idea to call |fit_all_models|
        % before running any of these (though you don't /have/ to).
        
        function plot_kernels(obj)
           for s_id=1:length(obj.sessions)
               fh = obj.sessions(s_id).plot_kernels();
               set(fh, 'name', join([sprintf("Session %d", s_id), fh.Name], " - "));
           end
        end
        
        function plot_predictions(obj)
           for s_id=1:length(obj.sessions)
               fh = obj.sessions(s_id).plot_predictions();
               set(fh, 'name', join([sprintf("Session %d", s_id), fh.Name], " - "));
           end
        end
        
        function plot_deviance_summary(obj)
           for s_id=1:length(obj.sessions)
               fh = obj.sessions(s_id).plot_deviance_summary();
               set(fh, 'name', join([sprintf("Session %d", s_id), fh.Name], " - "));
           end
        end
        
        function plot_tuning_summary(obj)
           for s_id=1:length(obj.sessions)
               fh = obj.sessions(s_id).plot_tuning_summary();
               set(fh, 'name', join([sprintf("Session %d", s_id), fh.Name], " - "));
           end
        end
        
        function plot_pathway_individual_kernels(obj, predictor_name, neuron_subset_aDMS, neuron_subset_pDMS)
            obj.get_pathway_populations();
            if nargin < 3
                neuron_subset_aDMS = [];
            end
            if nargin < 4
                neuron_subset_pDMS = [];
            end
            obj.populations('aDMS').plot_individual_kernels(predictor_name, neuron_subset_aDMS);
            obj.populations('pDMS').plot_individual_kernels(predictor_name, neuron_subset_pDMS);
        end
        
        function plot_pathway_choice_outcome_mixed_tuning(obj, threshold)
            %PLOT_PATHWAY_CHOICE_OUTCOME_MIXED_TUNING Compare tuning to
            %choice and choice x outcome on a neuron-by-neuron basis,
            %across pathways.
            %
            % If given, |threshold| draws a dashed reference line for both
            % quantities. For instance, calling this method with
            % threshold=2 will make it easy to tell by eye how many neurons
            % exceed a 2% tuning threshold for either choice or choice x
            % outcome.
            
            if nargin < 2
                threshold = NaN;
            end
            
            figure('Position', [670, 470, 900, 400], 'Name', 'Mixed choice/choice x outcome tuning comparison');
            hold on;
            popnames = {'aDMS', 'pDMS'};
            dummies = [NaN, NaN];
            for p=1:2
                popname = popnames{p};
                pop = obj.populations(popname);
                pop.plot_choice_outcome_mixed_tuning();
                dummies(p) = plot(NaN, NaN, 'o', 'Color', obj.colors_lines(popname));
            end
            xlabel('Choice tuning (%)')
            ylabel('Choice x Outcome tuning (%)')
            
            xl = xlim;
            yl = ylim;
            plot(xl, [threshold, threshold], 'LineWidth', 0.5, 'LineStyle', ':', 'Color', 'k');
            plot([threshold, threshold], yl, 'LineWidth', 0.5, 'LineStyle', ':', 'Color', 'k');
            xlim(xl);
            ylim(yl);
            
            
            legend(dummies, popnames);
        end
        
        function plot_pathway_tuning_sparseness(obj, use_groups, tuning_threshold)
            %PLOT_PATHWAY_TUNING_SPARSENESS summarize the tuning sparseness
            %metric in aDMS vs in pDMS.
            %
            % Note that this is a pathway-level summary of a cell-level
            % measure.
            if nargin < 2
                use_groups = false;
            end
            if nargin < 3
                tuning_threshold = 0;
            end
            
            obj.get_pathway_populations;
            p_pDMS = obj.populations('pDMS').get_tuning_sparseness([], use_groups, tuning_threshold);
            p_aDMS = obj.populations('aDMS').get_tuning_sparseness([], use_groups, tuning_threshold);
            
            figure('Position', [800,1200,460,260], 'Name', 'Tuning sparseness');
            
            % swarmplot
            subplot(1,2,1)
            hold on;
            
            scatter(ones(1,length(p_aDMS))+0.5*(rand(1,length(p_aDMS))-0.5),...
                p_aDMS, 'MarkerFaceColor', obj.colors('aDMS'), 'MarkerEdgeColor', 'w');
            scatter(0.8+ones(1,length(p_pDMS))+0.5*(rand(1,length(p_pDMS))-0.5),...
                p_pDMS, 'MarkerFaceColor', obj.colors('pDMS'), 'MarkerEdgeColor', 'w');
            xticks([1,2])
            xlim([0.48, 2.32])
            xticklabels({'aDMS', 'pDMS'})
            ylabel('Tuning sparseness')
            
            % histogram
            subplot(1,2,2)
            hold on;
            binwidth = 0.1;
            binedges = 0:binwidth:1;
            bincenters = (binedges(2:end)+binedges(1:end-1))/2;
            
            freqs_aDMS = histcounts(p_aDMS, binedges)/length(p_aDMS);
            freqs_pDMS = histcounts(p_pDMS, binedges)/length(p_pDMS);
            barh(bincenters, -freqs_aDMS, 1,...
                'FaceColor', obj.colors('aDMS'), 'EdgeColor', obj.colors_lines('aDMS'))
            barh(bincenters, freqs_pDMS, 1,...
                'FaceColor', obj.colors('pDMS'), 'EdgeColor', obj.colors_lines('pDMS'))
            xticks([-0.2, 0, 0.2])
            xticklabels([0.2, 0, 0.2])
            ylim([0.4,1])
            xlim([-max(xlim), max(xlim)]);
            text(-0.23,0.45,'aDMS','Color', obj.colors('aDMS'), 'HorizontalAlignment', 'left');
            text(0.23,0.45,'pDMS','Color', obj.colors('pDMS'), 'HorizontalAlignment', 'right');
            xlabel('Cell fraction')
            
            fprintf("Rank-sum test for equal medians of sparseness: %f\n", ranksum(p_aDMS, p_pDMS));
        end
        
        function plot_pathway_tuning(obj, plot_total_and_groups)
            %PLOT_PATHWAY_TUNING plot population tuning summary contrasting
            %the aDMS and the pDMS pathways.
            %
            %   Note that this requires each session in the experiment to
            %   have a specified |pathway|.
            
            if nargin < 2
                plot_total_and_groups = true;
            end
           
            % load all populations defined for this experiment, but discard
            % all those that are not named either 'aDMS' or 'pDMS'.
            all_pops = obj.get_pathway_populations();
            pathway_names = {'aDMS', 'pDMS'};
            pops = containers.Map();
            for p_id=1:length(pathway_names)
                name = pathway_names{p_id};
                pops(name) = all_pops(name);
            end
            
            bar_scale = 1/3;
            n_pops = length(pops);
            
            figure('Position', [800, 1200, 600, 300], 'Name', 'Pathway tuning');
            hold on
            
            max_tuning = 0;
            max_ci = 0;
            for p_id=1:n_pops
                pathway = pathway_names{p_id};
                pop = pops(pathway);
                [tuning, tuning_ci] = pop.get_tuning_to_each_predictor();
                
                if ~plot_total_and_groups
                    n_pred_groups = length(obj.model.design_matrix_config.special_predictor_sets);
                    tuning = tuning(:,2:end-n_pred_groups);
                    tuning_ci = tuning_ci(:,2:end-n_pred_groups);
                end
                
                max_tuning = max(max_tuning, max(tuning.Variables));
                max_ci = max(max_ci, max(max(tuning_ci.Variables)));
                n_tuning = width(tuning);
                
                x_pos = (1:n_tuning)+bar_scale*(p_id-1-(n_pops-1)/2)/((n_pops-1));
                bar(x_pos, tuning.Variables, 'FaceColor', obj.colors(pathway), 'EdgeColor', obj.colors_lines(pathway), 'BarWidth', bar_scale/(n_pops-1));
                plot(repmat(x_pos, 2, 1), tuning_ci.Variables, 'Color', obj.colors_lines(pathway), 'LineWidth', 3);
            end
            
            xticks(1:n_tuning)
            ticknames = cellfun(@(s) obj.sessions(1).neurons(1).predname2printname(obj.sessions(1).neurons(1).varname2predname(s)), tuning.Properties.VariableNames);
            xticklabels(ticknames)
            xtickangle(30)
            ylabel('Pathway tuning (%)')
            
            if max_ci>1.5*max_tuning
                ylim([0,1.2*max_tuning])
            end
            
            dummy_handles = zeros(1, length(pathway_names));
            for p_id=1:n_pops
                pathway = pathway_names{p_id};
                dummy_handles(p_id) = bar(NaN, NaN, 'FaceColor', obj.colors(pathway), 'EdgeColor', obj.colors_lines(pathway));
            end
            legend(dummy_handles, pathway_names, 'Location', 'north');
        end

       function [fh, tests, pvalues] = plot_pathway_population_kernels(obj, parallel_workers, alph, dm_config, neuron_subset_aDMS, neuron_subset_pDMS)
           %PLOT_PATHWAY_POPULATION_KERNELS compare mean-displacement
           %population 'kernels' across the two pathways
           if nargin < 5
               neuron_subset_aDMS = [];
           end
           if nargin < 6
               neuron_subset_pDMS = [];
           end
           if nargin < 4
               dm_config = obj.model.design_matrix_config;
           end
           if nargin < 3
               alph = 0.05;
           end
           if nargin < 2
               parallel_workers = [];
           end
           
           obj.get_pathway_populations();
           fh = obj.populations('aDMS').plot_population_kernels([], dm_config, parallel_workers, neuron_subset_aDMS);
           obj.populations('pDMS').plot_population_kernels(fh, dm_config, parallel_workers, neuron_subset_pDMS);
           
           [tests, lags, pvalues] = obj.pathway_kernel_difference_test(dm_config, parallel_workers, alph);
           predictor_names = tests.keys;
           n_predictors = length(predictor_names);
           side = ceil(sqrt(n_predictors));
           for p_id = 1:n_predictors
               predictor_name = predictor_names{p_id};
               subplot(side, side, p_id);
               yloc = ylim();
               yloc = yloc(2);
               vals = double(tests(predictor_name));
               vals(vals==0) = NaN;
               vals(vals==1) = yloc;
               plot(lags(predictor_name), vals, 'LineWidth', 3, 'Color', obj.color_kernel_difference_test);
           end
           
           set(fh, 'name', "Mean-displacement population kernels");
           
       end
    end
    
    methods (Static)
        function [test, pvalue] = kernel_diff_bootstraptest(pop1_kernels, pop2_kernels, nboot, alph, opts)
            %KERNEL_DIFF_BOOTSTRAPTEST significance test for the difference
            %between kernel pathways, based on BCa bootstrap. See Efron,
            %section 15.4.
            %
            %   pop1_kernels is an n_neurons x n_timepoints matrix
            %   containing the fitted kernels for a particular predictor
            %   for all cells in population 1.
            %
            %   pop2_kernels is the same, for population 2.
            %
            %   nboot is the number of bootstrap samples to use.
            %
            %   alph is the significance threshold.
            true_diff = rms(pop1_kernels) - rms(pop2_kernels);
  
            % compute bootstrap distributions (this is equivalent to doing
            % stratified bootstrap on the joint dataset with both pop1 and
            % pop2)
            boot_1 = bootstrp(nboot, @rms, pop1_kernels, 'Options', opts);
            boot_2 = bootstrp(nboot, @rms, pop2_kernels, 'Options', opts);
            boot_diff = boot_1 - boot_2;
            
            % compute jackknife distributions
            N1 = size(pop1_kernels,1);
            theta1 = zeros(size(pop1_kernels));
            for n=1:N1
                theta1(n,:) = rms(pop1_kernels([1:n-1 n+1:end],:));
            end
            theta1_dot = mean(theta1,1);
            U1 = (N1-1)*(theta1_dot-theta1);
            
            N2 = size(pop2_kernels,1);
            theta2 = zeros(size(pop2_kernels));
            for n=1:N2
                theta2(n,:) = rms(pop2_kernels([1:n-1 n+1:end],:));
            end
            theta2_dot = mean(theta2,1);
            U2 = (N2-1)*(theta2_dot-theta2);
            
            % compute acceleration constant (Eq. 15.36 in Efron)
            ahat = (1/6)*(sum(U1.^3,1)/N1^3 + sum(U2.^3,1)/N2^3)./(sum(U1.^2,1)/N1^2 + sum(U2.^2,1)/N2^2).^(3/2);
            
            % formula for the lags where true_diff>0
            alpha0p = sum(boot_diff<0, 1)/nboot;
            w0p = norminv(alpha0p);
            zhat0p = norminv(sum(boot_diff<true_diff, 1)/nboot);
            pvaluep = normcdf((w0p-zhat0p)./(1+ahat.*(w0p-zhat0p))-zhat0p);
            
            % formula for the lags where true_diff<0
            alpha0m = sum(boot_diff>0, 1)/nboot;
            w0m = norminv(alpha0m);
            zhat0m = norminv(sum(boot_diff>true_diff, 1)/nboot);    
            pvaluem = normcdf((w0m-zhat0m)./(1+ahat.*(w0m-zhat0m))-zhat0m);
            
            pvalue = pvaluep;
            pvalue(true_diff<0) = pvaluem(true_diff<0);
            test = pvalue <= alph;
        end 
        
    end
    
    
    %% ----PRIVATE INTERFACE----
    % These are methods that are only supposed to be used by other
    % components of the library.
    methods (Access=private)
        
        function clear_dms_and_fits(obj)
            %CLEAR_DMS_AND_FITS clear any existing design matrix or model
            %fit from all sessions.
            for s_id=1:length(obj.sessions)
                obj.sessions(s_id).clear_design_matrix();
                for n=1:length(obj.sessions(s_id).neurons)
                    obj.sessions(s_id).neurons(n).clear_fits();
                end
            end
        end
        
    end

end
