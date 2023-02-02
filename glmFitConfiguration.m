classdef glmFitConfiguration
    %GLMFITCONFIGURATION Configuration for fitting and evaluating GLM
    %  This should act as a simplified wrapper for glmnet, and also contain
    %  miscellaneous settings about how we look at the fitted model.
    
    properties
        %---------------------------
        % settings that will be applied directly through glmnetSet:
        %---------------------------
        
        % alpha - interpolation between Ridge (0) and Lasso (1).
        alpha = 0.95
        
        % intr - should intercept(s) be fitted (true) or set to zero
        % (false). BE CAREFUL THOUGH if you set this to false, as "glmnet
        % for matlab" gives a wrong FDE in that case, due to it using the
        % wrong null model. It seems like it is still using the intercept
        % to define the null, rather than taking the null to be the model
        % that predicts always zero. Note that recent versions of the
        % reference R implementation of glmnet do the right thing.
        intr = true
        
        % thresh - Convergence threshold for coordinate descent used in
        % glmnet, as documented in the glmnet documentation.
        % Note that the default here (1e-4) is that used in glmnet for
        % matlab. The reference R implementation, at least in more recent
        % versions, uses a much smaller value (1e-7). Modify as needed.
        thresh = 1e-4;
        
        % lambda - The number of lambda values
        nlambda = 100;
        
        
        %---------------------------
        % other settings
        %---------------------------
        
        % n_folds - number of crossvalidation folds for tuning of
        % regularization strength. The larger this number is, the longer
        % the fits will take, but (up to a point) the more stable the fits
        % should be. If the fits are unstable (i.e. if running the same fit
        % twice gives different kernels), try increasing n_folds.
        n_folds = 50
        
        % lambda_selection_criterion - this is used to select a final
        % value of the regularization strength when extracting kernels,
        % making predictions, or computing the FDE.
        lambda_selection_criterion = 'lambda_1se'
        
        % method - method to use to perform the fit. Can be
        % 'gmlnet_matlab', 'glmnet_R', 'gglasso' or 'SGL'. Note that
        % with gglasso and SGL not all functionality is implemented.
        %
        % Whatever your operating system, if you are going to use an
        % R-based package (i.e., anything but glmnet-matlab) remember to
        % make sure that R is on your PATH. From within matlab, this can be
        % done as follows:
        %
        % Linux/Mac OS: something like
        % 
        % setenv('PATH', ['/path/to/R/bin:', getenv('PATH')]);
        %
        % Windows: something like
        % 
        % setenv('PATH', ['C:\Program Files\R\R-4.0.2\bin;', getenv('PATH')]);
        method = 'glmnet_matlab';
        
        % skip_zero_kernel_fde - boolean flag to control whether reduced
        % models should be fit for those predictors that have identically
        % zero kernels in the full model, when computing the reduced FDE.
        % If true, reduced models will not be fit and the FDE in the
        % reduced model will be assumed to be equal to the FDE of the full
        % model. This saves time. Default: true. It can be set to false if
        % you want to perform a stability check on the fit, for instance
        % when assessing the number of CV folds to use (the idea being that
        % if the fit is unstable, the reduced model could have a lower FDE
        % of the full model even if the kernel in the full model was zero).
        skip_zero_kernel_fde = true;
    end
    
    properties (SetAccess = immutable)
        % parallel - as the |parallel| parameter passed to cvglmnet.
        % CURRENTLY ONLY SUPPORTED UNDER THE R IMPLEMENTATION OF GLMNET due
        % to a bug in the "glmnet for matlab" package.
        %
        % Here we set it permanently to false as it turns out it doesn't
        % help much for our analysis, and we are better off parallelizing
        % at a different level (across neurons).
        parallel = false;
    end
end

