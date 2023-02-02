function [choiceProbabilities,weights,rpe] = LV_QLearn_SoftmaxDecay_2CSR(SessionData,alpha,beta,decay)
% Here, the unchosen port decays according to the decay rate. 

    if ~exist('SessionData','var')
        uiopen 
    end
    
    [choices,rewards]=extractChoices_2CSR(SessionData);
    weights = zeros(2,SessionData.nTrials);      %assume 0 for starting conditions (could fit)
    rpe = zeros(size(weights)); 

    %% Train Weights
    for n = 1:SessionData.nTrials-1
        %compute rpe
        switch choices(n)
            case 1
            rpe(1,n) = (rewards(1,n)) - weights(1,n); %compute rpe (negative rpe for 0uL rewards
            rpe(2,n) = (0) - weights(1,n);
            weights(1, n+1) = weights(1,n) + alpha*rpe(1,n);     %update chosen
            weights(2, n+1) = weights(2,n) + decay*rpe(2,n);     % update unchosen with 0
            case 2
            rpe(1,n) = (0) - weights(2,n); %compute rpe (negative rpe for 0uL rewards
            rpe(2,n) = rewards(2,n) - weights(2,n);
            weights(1, n+1) = weights(1,n) + decay*rpe(1,n);     %update chosen
            weights(2, n+1) = weights(2,n) + alpha*rpe(2,n);     % update unchosen with 0
        end
        if weights(1, n+1)<0
            weights(1, n+1)=0;
        end
        if weights(2, n+1) <0
            weights(2, n+1)=0;
        end
    end
      
    choiceProbabilities = zeros(2,SessionData.nTrials);
    
     for i=1:SessionData.nTrials
       choiceProbabilities(1,i)= 1/...
           ( 1+exp(1)^...
           -(beta*(weights(1,i)-weights(2,i))) );

       choiceProbabilities(2,i)= 1/...
           ( 1+exp(1)^...
           -(beta*(weights(2,i)-weights(1,i))) );

     end
    
end