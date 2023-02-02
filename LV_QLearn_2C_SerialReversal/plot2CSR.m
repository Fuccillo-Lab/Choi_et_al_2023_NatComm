function plot2CSR(result)
%%
    [choices,rewards]=extractChoices_2CSR(result.SessionData);
    whichModel=result.model;
    SessionData=result.SessionData;
    maxReward=result.SessionData.TrialSettings(1).GUI.RewardAmountLarge;
    scatterSize=SessionData.nTrials/30;
    unrewardedTrials=find(cell2mat(SessionData.RewardVolume)==0);
        figure()
        hold on;
%         plot(1:SessionData.nTrials,(rewards(1,1:SessionData.nTrials)),'Color','b','LineWidth',3);
%             plot(1:SessionData.nTrials,(rewards(2,1:SessionData.nTrials)),'Color','r','LineWidth',3);
        xlabel('Trial Number')
        ylabel('Q Value')
        ylim([0 maxReward+1])

        legend({'Left Choice','Right Choice','LeftQValue','RightQValue'},'AutoUpdate','off','Location','southwest')
        legend('boxoff')
        history1=find(choices==1);
        history2=find(choices==2);
        scatter(history1,ones(1,length(history1))*maxReward+0.2,scatterSize,'b','filled')
        scatter(history2,ones(1,length(history2))*maxReward+0.2,scatterSize,'y','filled')
        scatter(unrewardedTrials,ones(1,length(unrewardedTrials))*maxReward+0.4,scatterSize,'s','k','filled')
% 

            plot(1:SessionData.nTrials,result.Qvalues(1,:),'-','Color','b','LineWidth',2.5);
            plot(1:SessionData.nTrials,result.Qvalues(2,:),'-','Color','y','LineWidth',2.5);
%         scatter(wcL,ones(1,length(wcL))*13,'b')
%         scatter(wcR,ones(1,length(wcR))*13,'g')
        
        legend({'Left Choice','Right Choice','Unrewarded Trial','LeftQValue','RightQValue'},...
            'AutoUpdate','off','Location','southeastoutside')
        legend('boxoff')
        str=['       Alpha: ',num2str(result.alpha)];
        if whichModel=='SoftDec'
            str=['Model: ',whichModel,'       Likelihood: ',num2str(result.likelihood)...
                '       Alpha: ',num2str(result.alpha), '       Beta: ' num2str(result.beta),...
                '       Decay: ' num2str(result.decay)];
        end
        
        if whichModel=='SoftMax'
            str=['Model: ',whichModel,'       Likelihood: ',num2str(result.likelihood)...
                '       Alpha: ',num2str(result.alpha), '       Beta: ' num2str(result.beta)];
        end
        
        if whichModel=='Epsilon'
            str=['Model: ',whichModel,'       Likelihood: ',num2str(result.likelihood)...
                '       Alpha: ',num2str(result.alpha), '       Beta: ' num2str(result.beta),...
                '       Epsilon: ' num2str(result.epsilon)];
        end
        
        if whichModel=='EpsiDec'
            str=['Model: ',whichModel,'       Likelihood: ',num2str(result.likelihood)...
                '       Alpha: ',num2str(result.alpha), '       Beta: ' num2str(result.beta),...
                '       Epsilon: ' num2str(result.epsilon), '       Decay: ' num2str(result.decay)];
        end
        
        text(SessionData.nTrials/5,maxReward+0.8,str)
end
    