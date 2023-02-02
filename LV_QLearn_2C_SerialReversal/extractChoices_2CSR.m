function [choices,rewards]=extractChoices_2CSR(SessionData)

if ~exist('SessionData','var')
   uiopen 
end

zero=cell(1);
zero{1}=0;

choices=SessionData.Choice;
choice=zeros(1,SessionData.nTrials);
rewards=SessionData.TrialTypes;
reward=zeros(2,SessionData.nTrials);
for i=1:SessionData.nTrials
    % Define what the animal's possible rewards will be
%        if length(cell2mat(rewards(i)))==length('right')
% %           reward(1,i)=SessionData.TrialSettings(1).GUI.RewardAmountSmall;
% %           reward(2,i)=SessionData.TrialSettings(1).GUI.RewardAmountLarge;
%           reward(1,i)=0;
%           reward(2,i)=8;
%        else
% %            reward(1,i)=SessionData.TrialSettings(1).GUI.RewardAmountLarge;
% %            reward(2,i)=SessionData.TrialSettings(1).GUI.RewardAmountSmall;
%            reward(1,i)=8;
%            reward(2,i)=0;           
%        end
       if SessionData.RewardVolume{(i)}==0
           reward(1,i)=0;
           reward(2,i)=0;
       else
           switch choices(i)
               case -1
                   reward(1,i)=SessionData.TrialSettings(1).GUI.RewardAmountLarge;
               case 1
                   reward(2,i)=SessionData.TrialSettings(1).GUI.RewardAmountLarge;
           end
       end
       
       % Make it so that choices are 1 for left and 2 for right
       if choices(i)==-1
           choice(i)=1;
       else 
           choice(i)=2;
       end
end

rewards=(reward);
choices=choice;
end


