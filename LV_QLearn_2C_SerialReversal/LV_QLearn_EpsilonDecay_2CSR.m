function [choiceProbabilities,weights,rpe] = LV_QLearn_EpsilonDecay_2CSR(SessionData,alpha,epsilon,decay)
    
    if ~exist('SessionData','var')
        uiopen 
    end
    
    [choices,rewards]=extractChoices_2CSR(SessionData);
    weights = zeros(2,SessionData.nTrials);  
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
    
    choiceProbabilities=zeros(2,SessionData.nTrials);
    % choice probabilities of 0 break the model, so they will be set to
    % a minimal value known as eps
    for i=1:SessionData.nTrials
        if rand(1)<=epsilon
            choiceProbabilities(1,i)=0.5;
            choiceProbabilities(2,i)=0.5;
        else
            if weights(1,i)>weights(2,i)
                choiceProbabilities(1,i)=1;
                choiceProbabilities(2,i)=eps;
            elseif weights(1,i)==weights(2,i)
                choiceProbabilities(1,i)=0.5;
                choiceProbabilities(2,i)=0.5;
            else
                choiceProbabilities(1,i)=eps;
                choiceProbabilities(2,i)=1;
            end
        end
    end
    
end