classdef designMatrixConfiguration < handle
    %DESIGNMATRIXCONFIGURATION Configuration for how to make design matrix
    %   This should contain the necessary information about which event
    %   types must be taken into account, and the parameters for the time
    %   expansion.
    
    properties (Constant)
        default_spline_degree = 3
        
        % special predictor sets which are interesting to manipulate
        % collectively
        special_predictor_sets = containers.Map({'internal', 'action', 'all_outcome'},...
            {["Q_value_1", "Q_value_-1", "Q_chosen",...
            "Q_value_1_Tinit_start", "Q_value_-1_Tinit_start", "Q_chosen_Tinit_start"...
            "Q_value_1_Tchoice", "Q_value_-1_Tchoice", "Q_chosen_Tchoice"...
            "Q_value_1_Toutcome", "Q_value_-1_Toutcome", "Q_chosen_Toutcome"...
            "Q_difference", "Q_sum", "Q_signed_difference_Toutcome",...
            "Q_signed_difference_Toutcome_pos", "Q_signed_difference_Toutcome_neg",...
            "Q_signed_difference_Tinit_start", "Q_signed_difference_Tinit_start_pos",...
            "Q_signed_difference_Tinit_start_neg", "RPE", "RPE_pos" "RPE_neg",...
            "state_Tchoice", "state_Tchoice_neg", "state_Tchoice_pos",...
            "state_Tinit_start", "state_Tinit_start_neg", "state_Tinit_start_pos",...
            "state_Toutcome", "state_Toutcome_neg", "state_Toutcome_pos",...
            "reward_rate"],... % predictors considered "internal"
            ["init_start", "choice_1", "choice_-1", "head_velocity"],... % predictors related to task initiation and action
            ["outcome", "outcome_1", "outcome_-1", "choice_-1:outcome_-1", ...
            "choice_-1:outcome_1", "choice_1:outcome_-1", "choice_1:outcome_1"]})
    end
    
    properties (SetAccess=private)
        config % table with the core information (predictor names, window size, number of knots, and spline degree)
        column_indices % map storing the column indices associated to each predictor
        column_names % list of the predictor name associated with each column
        column_counter % counter that keeps track of how many columns are in the DM
        excluded_predictors % list of predictors currently excluded from the model (useful for building reduced models)
    end
    
    properties (SetAccess={?session})
         % frozen - a boolean flag that is set to true the first time a
         % concrete design matrix is generated from this configuration.
         % This is used to prevent users from altering the configuration
         % after it has already been used, which could result in
         % inconsistencies.
        frozen = false
    end
    
    properties
        use_special_predictor_groups = true % whether to make use of the special groups when computing and plotting tuning
    end
    
    methods
        function obj = designMatrixConfiguration()
            %DESIGNMATRIXCONFIGURATION Construct an instance of this class
            obj.config = table();
            obj.column_indices = containers.Map;
            obj.column_names = strings(0);
            obj.column_counter = 1;
            obj.excluded_predictors = strings(0);
        end
        
        function newConfig = copy(obj, preserve_frozen)
            %COPY return a new, unfrozen copy of this DM configuration.
            newConfig = designMatrixConfiguration();
            newConfig.config = obj.config;
            newConfig.column_indices = obj.column_indices;
            
            newConfig.column_counter = obj.column_counter;
            newConfig.excluded_predictors = obj.excluded_predictors;
            if nargin==2 && preserve_frozen
                newConfig.frozen = obj.frozen;
            end
        end
        
        
        function add_predictor(obj, predictor_name, n_knots, window_pre, window_post, degree)
            %ADD_PREDICTOR Add the settings relative to one predictor.
            %
            %
            %   |predictor_name| : the desired name for the predictor.
            %
            %   |n_knots| : generally this indicated the number of knots to
            %   use for the spline basis set. There are a couple of
            %   exceptions to this though, designed to allow one to
            %   configure predictors that are not kernel-based. If
            %   |n_knots| is zero, we assume "stepwise" coding of the
            %   predictor, which means that the predictor will stay
            %   constant at the value of its associated event until the
            %   last timepoint preceding that event following the event,
            %   and will then switch to the value of the next event of the
            %   same type immediately after. This type of predictor will
            %   take up one column of the design matrix. A negative value
            %   of |n_knots| is only meant to be used in conjunction with
            %   the special predictor 'autoregressive' or 'reward_rate'. In
            %   this case, the absolute value of |n_knots| indicates the
            %   order of the autoregressive term, or the number of trials
            %   in the past to use to determine the running average of the
            %   reward rate. For instance, for the reward rate predictor if
            %   |n_knots|=-5 then the reward rate will be integrated over
            %   the last 5 trials. Analogously, for the autoregressive case
            %   if |n_knots|=-2 then we will have two columns in the design
            %   matrix for the autoregressive term, one containing the
            %   neuron's activity at time t-1, and the other the activity
            %   at time t-2. If |predictor_name|='head_velocity', this
            %   argument will be ignored.
            %
            %   |window_pre| : for kernel-based predictors, the size of the
            %   pre-event side of the window in milliseconds. For instance,
            %   if |window_pre|=1500, then the kernel window for this event
            %   will open 1.5s before each event of the associated type.
            %
            %   |window_pre| : for kernel-based predictors, the size of the
            %   post-event side of the window in milliseconds. For
            %   instance, if |window_post|=2000, then the kernel window for
            %   this event will close 2s after each event of the associated
            %   type.
            
            if obj.frozen
                error("you can't modify a design matrix configuration after it has already been used to create a concrete DM. Please create a new DM (possibly using the copy() method) and pass it to experiment.set_design_matrix().")
            else
                if nargin==2
                    n_knots = 0;
                end
                if n_knots > 0 && nargin == 5
                    % this is the case of a regular event-based predictor with
                    % unspecified spline degree, so we use the default
                    degree = obj.default_spline_degree;
                end
                if n_knots <= 0
                    % this must be a continuous predictor (currently, this
                    % can be either a reward rate, and autoregressive term
                    % or a stepwise term)
                    window_pre = 0;
                    window_post = 0;
                    degree = 0;
                    if n_knots == 0
                        % this is a stepwise term
                        n_columns = 1;
                    else
                        % autoregressive or reward rate
                        n_knots = abs(n_knots);
                        if any(strcmp(predictor_name, {'reward_rate', 'head_velocity'}))
                            n_columns = 1;
                        else
                            n_columns = n_knots;
                        end
                    end
                else
                    n_columns = n_knots + 2 + degree - 1;
                end
                % check that window sizes are not negative
                assert(window_pre>=0 && window_post>=0, "Attempted to set negative window size.")
                % check that if we are adding an interaction term, the main
                % effects have been already included
                if contains(predictor_name, ":")
                    main_effects = string(split(predictor_name, ":"));
                    assert(length(main_effects)==2, "Three-way interactions are not supported.")
                    for e_id=1:length(main_effects)
                        main_effect = main_effects(e_id);
                        if ~any(strcmp(main_effect, obj.get_active_predictors()))
                            error(...
                                "trying to add interaction %s, but the main effect %s is missing",...
                                predictor_name,...
                                main_effect);
                        end
                    end
                end
                
                
                temp = struct2table(struct(...
                    'window_pre', window_pre,...
                    'window_post', window_post,...
                    'n_knots', n_knots,...
                    'degree', degree),...
                    'RowNames', {predictor_name});
                obj.column_indices(predictor_name) = obj.column_counter:obj.column_counter + n_columns - 1;
                obj.column_names = [obj.column_names, repmat(string(predictor_name), 1, n_columns)];
                obj.column_counter = obj.column_counter + n_columns;
                obj.config = [obj.config; temp];
                
            end
        end
        
        function indices = get_group_indices(obj)
            %GET_GROUP_INDICES Useful for SGL
            active_predictors = obj.get_active_predictors();
            indices = [];
            for p_id = 1:length(active_predictors)
                predictor = active_predictors(p_id);
                n_cols = nnz(obj.column_names==predictor);
                indices = [indices, repmat(p_id, 1, n_cols)];
            end
        end
        
    end
    
    
    methods (Access={?session,?neuron,?population})
        
        function obj = exclude_predictors(obj, predictor_names)
            %EXCLUDE_PREDICTORS Mark one or more predictors as "excluded".
            
            n_predictors = length(predictor_names);
            for p_id=1:n_predictors
                assert(any(strcmp(obj.config.Properties.RowNames, predictor_names(p_id))),...
                    "You can only exclude predictors that are present in the full design matrix.");
            end
            
            obj.excluded_predictors = [obj.excluded_predictors, predictor_names];
        end
        
        function reducedConfig = get_reduced_variant(obj, predictors_to_exclude)
            %GET_REDUCED_VARIANT Build a config variant without the given
            %predictor(s).
            %
            %   |predictors_to_exclude| can be a cell array of char or a
            %   list of strings.
            %
            %   This is meant to be used when building reduced variants of
            %   a "full" encoding model.
            %
            %   This operation is nontrivial in the case of interaction
            %   predictors. For instance, suppose that you have X1, X2 and
            %   X1:X2 in your current DM config. If you want the reduced
            %   model without the interaction term, there is no problem,
            %   you just remove it from the model. But if you want to
            %   exclude X1, you now have to exclude X1:X2 too, otherwise
            %   X1:X2 may absorb part of the deviance that's captured by X1
            %   alone.
            
            predictors_to_exclude_expanded = strings(0);
            
            for p_id=1:length(predictors_to_exclude)
                % for each predictor to be excluded, figure out what other
                % predictors are "related" interaction predictors, in the
                % sense that their name contains both a colon (indicating
                % that they are an interaction) and the name of the
                % predictor to be excluded. Exclude those too.
                predictor = predictors_to_exclude(p_id);
                related_interactions = obj.get_interactions_from_main_effect(predictor);
                predictors_to_exclude_expanded = [predictors_to_exclude_expanded, predictor, related_interactions];
            end
            reducedConfig = obj.copy(true).exclude_predictors(predictors_to_exclude_expanded);
%             %DEBUG
%             fprintf("REDUCED: Asked to exclude predictors %s, going to exclude predictors %s\n",...
%                 strjoin(predictors_to_exclude), strjoin(predictors_to_exclude_expanded));
        end
        
        function active_predictors = get_active_predictors(obj)
            
            all_names = obj.config.Properties.RowNames;
            active_predictors = strings(0);
            for n_id=1:length(all_names)
                predictor_name = all_names(n_id);
                if ~any(strcmp(obj.excluded_predictors, predictor_name))
                    active_predictors = [active_predictors, predictor_name];
                end
            end   
        end
        
        function predictors = get_special_predictor_set(obj, set_name)
            %GET_SPECIAL_PREDICTOR_SET get the list of predictor names
            %currently active that belong to one of the special predictor
            %sets.
            predictors = intersect(obj.get_active_predictors(),...
                obj.special_predictor_sets(set_name));
        end
        
        function name = get_model_name(obj)
            % GET_MODEL_NAME name for choice of config+excluded predictors
            %    This is useful when storing multiple variants of data (for
            %    instance fit objects or FDE values) that are computed for
            %    the full model as well as for the reduced variants.
            if isempty(obj.excluded_predictors)
                name = 'full';
            else
                name = char(join(obj.excluded_predictors, "+"));
            end 
        end
        
        function related_interactions = get_interactions_from_main_effect(obj, predictor)
            %GET_INTERACTIONS_FROM_MAIN_EFFECT if a predictor is a main
            %effect, that is if there are active interaction terms built
            %from it AND with the same window, return the interaction
            %terms.
            active_predictors = obj.get_active_predictors();
            interactions = active_predictors(contains(active_predictors, ":"));
            
            % make sure |predictor| is char, as we will have to use it to
            % row-index into a table below
            predictor = char(predictor);
            
            related_interactions = strings(0);
            
            for i_id=1:length(interactions)
                interaction = interactions(i_id);
                main_effects = string(split(interaction, ":"));
                if ismember(predictor, main_effects) && ...
                        ~ismember(interaction, related_interactions) && ...
                        all(obj.config{char(interaction),{'window_pre', 'window_post'}} == obj.config{predictor,{'window_pre', 'window_post'}})
                    related_interactions = [related_interactions, interaction];
                end
            end            
        end
        
    end
    
end

