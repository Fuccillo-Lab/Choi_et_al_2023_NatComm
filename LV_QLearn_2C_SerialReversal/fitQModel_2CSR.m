function [result] = fitQModel_2CSR(SessionData,model)
    % Inputs: 
    % 1. Session Data is 2-choice serial reversal session data from BPod 
    % 2. Model is a 5-letter code for the type of choice algorithm to use.
    % The possible choices are as follows:
    
    % 'SoftMax' = a softmax model where choice probability is somewhat proportional
    % to the Q Value of the choice. Proportionality depends on beta
    % parameter. High beta exploits value differences, low beta does not.
    % 'SoftDec' = a softmax model where the value of the choice that isn't
    % chosen decays according to the decay parameter. There is an RPE for unchosen choice
    % 'Epsilon' = a choice algorithm where some percentage of the time a
    % choice is made randomly, and the other times the choice is made
    % according to highest Q value. Epsilon Parameter controls how often a
    % random choice is made (epsilon = 0.5 is random half of the time)
    % EpsiDec = a greedy-epsilon model where the unchosen choice decays in
    % value according to the decay parameter
     
    
    % Outputs: Result Matrix with Fitted Model parameters,
    % Trial-by-trial RPEs & Qvalues, & Model Likelihood
    
    % 1. Alpha is the learning rate. 
    % 2. Beta is the explore/exploit parameter.
    %    Beta < 1 is exploratory, Beta > 1 is exploitative.
    %    If testing epsilon model, beta will be set to 1.
    % 3. Epsilon is the percentage of trials where a random choice is made.
    %    If testing a softmax algorithm, epsilon will be set to 0.
    % 4. Decay is the fitted decay rate for models that include decay. Decay 
    %    will be set to 0 for models that don't include decay rates.
    % 5. Choice Probabilities is the probability of selecting each choice
    %    according to the Q value and beta parameter.
    % 6. RPEs is a matrix containing trial-by-trial RPEs for each choice. 
    %    Depending on the model used, there will or won't be an RPE for
    %    each choice
    % 7. QValues is a matrix containing the trial-by-trial Q value of each
    %    choice. Top row is for the left choice, bottom row is for right
    %    choice.
    % 8. QSums is the trial-by-trial sum of the Q value for each choice
    % 9. QDifferences is the trial-by-trial absolute difference between
    %    each of the choices
    % 10. Likelihood is a measure of the likelihood of the model. Higher is
    %    better.
    
    % Coded by Luigim Vargas. June 29, 2020.

%% Initiate parameters for fitting
    Aeq=[];
    beq=[];
    Aineq=[];
    bineq=[];
    lb = [0 0 0 0];    % lower limits for alpha, beta, epsilon, and decay rate
    ub = [1 100 1 1];  % upper limits for alpha, beta, epsilon, and decay rate
    inx = [0.5 0.5 0.5 0.5]; % starting points to fit from (shouldn't matter)

    options = optimset('Display','on','MaxIter',5000000,'TolFun',1e-15,'TolX',1e-15,...
        'DiffMaxChange',1e-2,'DiffMinChange',1e-6,'MaxFunEvals',5000000,...
        'LargeScale','off');
    %% Define algorithm and options for fitting
    tester = @(inputs)compareModelFit_2CSR(SessionData, inputs(1), 0, model,...
        inputs(2), inputs(3),inputs(4));
    problem = createOptimProblem('fmincon','objective',tester,'x0', inx,...
        'lb',lb,'ub',ub,'Aeq',Aeq,'beq',beq,'Aineq',Aineq,'bineq',bineq,'options',options);
    ms = MultiStart;
    k = 100;

    warning off;
    %% Fit parameters according to the model and behavioral data
    [inputs, loglike, exitflag, output, solutions] = run(ms,problem,k);
    %% Create output for results
    
    if model=='Epsilon'
        inputs(2)=1;
        inputs(4)=0;
        [choiceProbabilities, Qvalues,RPEs]=LV_QLearn_Epsilon_2CSR(SessionData,...
        inputs(1),inputs(3));
    end
    if model=='SoftMax' 
        inputs(3)=0;
        inputs(4)=0;
        [choiceProbabilities, Qvalues,RPEs]=LV_QLearn_Softmax_2CSR(SessionData,...
        inputs(1),inputs(2));
    end
    if model=='EpsiDec'
        inputs(2)=1;
        [choiceProbabilities, Qvalues,RPEs]=LV_QLearn_EpsilonDecay_2CSR(SessionData,...
        inputs(1),inputs(3),inputs(4));
    end
    if model=='SoftDec' 
        inputs(3)=0;
        [choiceProbabilities, Qvalues,RPEs]=LV_QLearn_SoftmaxDecay_2CSR(SessionData,...
        inputs(1),inputs(2),inputs(4));
    end
    
    QSums=zeros(1,SessionData.nTrials);
    QDiffs=zeros(1,SessionData.nTrials);
    for i=1:SessionData.nTrials
        QSums(i)=Qvalues(1,i)+Qvalues(2,i);
        QDiffs(i)=abs(Qvalues(1,i)-Qvalues(2,i));
    end
    
    [choices,rewards]=extractChoices_2CSR(SessionData);
    
    result.model = model;
    result.alpha = inputs(1);
    result.beta = inputs(2);
    result.epsilon = inputs(3);
    result.decay = inputs(4);
    result.choiceProbabilities = choiceProbabilities;
    result.RPEs = RPEs;
    result.Qvalues = Qvalues;
    result.QSums = QSums;
    result.QDifferences = QDiffs;
    result.likelihood = loglike; 
    result.choices = choices;
    result.rewards = rewards;
    result.SessionData = SessionData;

end