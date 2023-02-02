%% Simulated 2CSR Task
% Luigim Vargas Cifuentes 2/17/2020
% rewardRate should be a value between 0 and 1 for percentage of rewarded
% trials, where 1=100%, 0.5=50%, 0.01=1%, etc.
% winStay and loseShift should be values between 0 and 1, decimals included


function[numberOfRewards]=Simulate2CSR(rewardProbability,winStay,loseShift)


%% Initiate trial conditions
chance = rand(1);

if chance<0.5
    rewSide=1;
else
    rewSide=2;
end
corrects=zeros(1,300);
lastSwitch=1;
%% Loop through trials
for trialN = 1:300
     if trialN==9
         cumulCorrect=sum(corrects(1,1:8));   
     elseif trialN==10
         cumulCorrect=sum(corrects(1,1:9));
     elseif trialN>10
         cumulCorrect=sum(corrects(1,trialN-10:trialN-1));
     end
     
     if trialN>8 %only look at cumulative correct if we've gone through
                 %at least 8 trials
         if cumulCorrect>7 && trialN-lastSwitch>7
             lastSwitch=trialN;
             switch rewSide
                 case 1
                     rewSide=2;
                 case 2
                     rewSide=1;
             end
         end
     end     


%% Make a choice
    if trialN==1 %if it's the first trial, choice is made randomly
        chance = rand(1);
        if chance<0.5
            choiceN=1;
        else
            choiceN=2;
        end
    else
        prevOutcome=corrects(1,trialN-1);
        switch prevOutcome
            case 0 %if previous was incorrect, loseShift if percentage hits
                if chance<loseShift
                    if prevChoice==1
                        choiceN=2;
                    else
                        choiceN=1;
                    end
                else
                    choiceN=prevChoice; %if percentage misses, then we loseStay
                end
            case 1 %if previous was correct, winStay if percentage hits
                if chance<winStay
                    choiceN=prevChoice; %if percentage hits, then we winStay
                else %if percentage misses, then we winShift
                    if prevChoice==1
                        choiceN=2;
                    else
                        choiceN=1;
                    end
                end                
        end
    end
    prevChoice=choiceN; %save the current choice as previous for the next trial
%% Evaluate outcome
    if rewSide==choiceN
        corrects(1,trialN)=1;
    end
    
     chance=rand(1);
     if chance > rewardProbability %if this trial is unrewarded, outcome=0
         corrects(1,trialN)=0;
     end
    
end

numberOfRewards=sum(corrects);

end