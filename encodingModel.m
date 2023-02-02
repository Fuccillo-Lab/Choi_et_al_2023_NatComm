classdef encodingModel < handle
    %ENCODINGMODEL Forward (encoding) model of neural recordings.
    %   An encoding model is an explicit description of how we interpret
    %   the experimental data. In practice, it contains the instructions
    %   for building design matrices as well as those for fitting and using
    %   the associated GLM. It also provides methods to build objects that
    %   depend on our modeling choices but not on the data, such as the
    %   representation of a kernel.
    %
    %   Note that here "model" is meant as an *abstract* model of the data,
    %   and therefore anything that depends on the actual data does not
    %   belong here. For instance, the concrete instantiations of design
    %   matrices belong in their corresponding |session| objects, and
    %   fitted models (as well as methods for fitting and prediction)
    %   belong inside |neuron| objects.
    
    properties
        experiment
        design_matrix_config
        glm_fit_config = glmFitConfiguration()
    end
    
    methods
        function obj = encodingModel(dm_config, glm_fit_config)
            %ENCODINGMODEL Define a model with configs for a DM and a GLM
            
            obj.design_matrix_config = dm_config;
            if nargin ==2
                obj.glm_fit_config = glm_fit_config;
            end
        end
        
        function [kernel, lags] = get_kernel(obj, dm_config, predictor_name, coefs)
            %GET_KERNEL Compute kernel for one predictor
            %    Given the name of the predictor and a vector of
            %    coefficients that matches the number of basis elements for
            %    that predictor, return a representation of the kernel
            %    (i.e. the specific combination of basis elements
            %    given by the coefficients) over a standard grid of points
            %    spanning the predictor's window.
            predictor_kernel_config = dm_config.config(predictor_name,:);
            degree = predictor_kernel_config.degree;
            knots = obj.get_default_knots(predictor_kernel_config);
           
            if strcmp(predictor_name, "autoregressive")
                lags = -1:-1:-predictor_kernel_config.n_knots;
                kernel = coefs;
            elseif predictor_kernel_config{1,'n_knots'}==0 || strcmp(predictor_name, "reward_rate")
                % this is the case of a scalar predictor, for instance like
                % the stepwise predictor, which jumps to a new value every
                % time its associate event occurs, but is otherwise
                % constant in time. So the concept of lags doesn't apply
                % here. Therefore, we conventionally define a constant
                % "kernel" which in this case goes from 1ms before the
                % event to the moment of the event. This conventional
                % kernel can be useful for plotting.
                lags = [0; 1];
                kernel = [coefs; coefs];
            else
                lags = obj.get_default_kernel_sampling_points(predictor_kernel_config);
                basis = bsplinebasis(degree, knots, lags);
                kernel = basis * coefs;
            end
        end
        
        %-----Plotting methods below-----
        
        function fh = plot_example_kernel(obj, predictor_name, coefs, ca)
            %PLOT_EXAMPLE_KERNEL simple visualization for a kernel
            %associated with the given predictor.
            %
            %   Note that this does not require any fitting, as it allows
            %   to specify directly an arbitrary vector of coefficients.
            
            % ensure coefs is a column vector
            coefs = reshape(coefs, [], 1);
            % compute the kernel
            [kernel, lags] = obj.get_kernel(obj.design_matrix_config, predictor_name, coefs);
            % make plot
            if nargin < 4
                figure_scale = 3; % length of long side of the figure in inches
                fh = figure('Units', 'inches', 'Position', [5, 5, figure_scale, figure_scale*2/(1+sqrt(5))]);
            else
                axes(ca);
                fh = gcf;
            end
            hold on;
            area(lags, kernel);
            yl = ylim;
            plot([0,0], yl, 'LineStyle', ':', 'Color', 'r')
            xlabel('Time (ms)')
        end
    end
    
    methods (Access={?experiment,?session,?neuron})
        
        function check=is_new_dm_config(obj, dm_config)
            %IS_MY_DM_CONFIG Check if a DM config is different from mine
            check = isempty(obj.design_matrix_config) ||...
                ~(isequal(dm_config.config, obj.design_matrix_config.config) &&...
                dm_config.frozen==obj.design_matrix_config.frozen);
        end
    end
        
    methods (Access={?session}, Static)
        
        function knots = get_default_knots(predictor_kernel_config)
            %GET_DEFAULT_KNOTS Generate basic array of knots for predictor
            %    predictor_kernel_config is meant to be a row of a design
            %    matrix configuration table. Note that n_knots indicates
            %    INTERNAL knots, so for instance if degree=3 and n_knots=5,
            %    there will be 5 internal knots plus (3+1) boundary knots
            %    at each end of the interval, for a total of 11 knots.
            
            n_knots = predictor_kernel_config.n_knots;
            window_pre = predictor_kernel_config.window_pre;
            window_post = predictor_kernel_config.window_post;
            degree = predictor_kernel_config.degree;
            
            % TODO: check that the window configuration is consistent with
            % how the basis is defined. If the window is bidirectional
            % (i.e., if both window_pre and window_post are larger than
            % zero), we require the number of knots to be odd, and we make
            % sure that the event corresponds to one node. If it is
            % one-directional, we don't care.
            
            knots = linspace(0, 1, n_knots+2);
            % add multiple boundary knots
            knots = [zeros(1, degree), knots, ones(1, degree)];
            
            window_length = window_pre + window_post;
            % scale to window length
            knots = knots * window_length;
            % shift so that the window is correctly split between pre and
            % post
            knots = knots - window_pre;
        end
        
    end
    
    methods (Access=?experiment, Static)
        
        function ts = get_default_kernel_sampling_points(predictor_kernel_config)
            %GET_DEFAULT_KERNEL_SAMPLING_POINTS Get standard grid of
            %    points where to evaluate a kernel to build a useful
            %    representation. This spans the entire event window.
            
            window_pre = predictor_kernel_config.window_pre;
            window_post = predictor_kernel_config.window_post;
           
            ts = (-window_pre:10:window_post)';
        end
    end

end

