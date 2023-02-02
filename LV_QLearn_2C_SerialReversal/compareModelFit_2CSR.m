%% Compare Model Fit
function [acc]=compareModelFit_2CSR(SessionData,alpha,doWePlot,whichModel,beta,epsilon,decay)
    if ~exist('SessionData','var')
       uiopen 
    end
    %% Check Model Fit    
    [choices,rewards]=extractChoices_2CSR(SessionData);
    weightedChoices = zeros(1,SessionData.nTrials);
    if whichModel=='Epsilon'
        [choiceProbabilities, weights,~]=LV_QLearn_Epsilon_2CSR(SessionData,alpha,epsilon);
        wcL=find(weightedChoices==1);
        wcR=find(weightedChoices==2);
        choices = choices - 1;%I'm more than a bit squeamish about the number of times we have changed L/R choices. 2 was right and 1 was left right? So I set 1 = right and 0 to left
        acc = -(transpose(choices(:))*log(choiceProbabilities(2,:))' + transpose((1-choices(:)))*log(choiceProbabilities(1,:))') ; %note that this is matrix multiplication so we get 1 values which will be a sum. See mtimes in matlab help
    elseif whichModel=='EpsiDec'
        [choiceProbabilities, weights,~]=LV_QLearn_EpsilonDecay_2CSR(SessionData,alpha,epsilon,decay);
        wcL=find(weightedChoices==1);
        wcR=find(weightedChoices==2);
        choices = choices - 1;%I'm more than a bit squeamish about the number of times we have changed L/R choices. 2 was right and 1 was left right? So I set 1 = right and 0 to left
        acc = -(transpose(choices(:))*log(choiceProbabilities(2,:))' + transpose((1-choices(:)))*log(choiceProbabilities(1,:))') ; %note that this is matrix multiplication so we get 1 values which will be a sum. See mtimes in matlab help
    elseif whichModel=='SoftMax'
        [choiceProbabilities, weights,~]=LV_QLearn_Softmax_2CSR(SessionData,alpha,beta);
        wcL=find(weightedChoices==1);
        wcR=find(weightedChoices==2);
        choices = choices - 1;%I'm more than a bit squeamish about the number of times we have changed L/R choices. 2 was right and 1 was left right? So I set 1 = right and 0 to left
        acc = -(transpose(choices(:))*log(choiceProbabilities(2,:))' + transpose((1-choices(:)))*log(choiceProbabilities(1,:))') ; %note that this is matrix multiplication so we get 1 values which will be a sum. See mtimes in matlab help
    elseif whichModel=='SoftDec'
        [choiceProbabilities, weights,~]=LV_QLearn_SoftmaxDecay_2CSR(SessionData,alpha,beta,decay);
        wcL=find(weightedChoices==1);
        wcR=find(weightedChoices==2);
        choices = choices - 1;%I'm more than a bit squeamish about the number of times we have changed L/R choices. 2 was right and 1 was left right? So I set 1 = right and 0 to left
        acc = -(transpose(choices(:))*log(choiceProbabilities(2,:))' + transpose((1-choices(:)))*log(choiceProbabilities(1,:))') ; %note that this is matrix multiplication so we get 1 values which will be a sum. See mtimes in matlab help
    end

    if doWePlot==true
        figure()
        hold on;
        plot(1:SessionData.nTrials,(rewards(1,1:SessionData.nTrials)),'Color','b','LineWidth',3);
            plot(1:SessionData.nTrials,(rewards(2,1:SessionData.nTrials)),'Color','g','LineWidth',3);
        xlabel('Trial Number')
        ylabel('uL of Sucrose')
        ylim([0 14])

%         legend({'Left Port','Center Port','Right Port','Unrewarded Trial','Exploit'},'AutoUpdate','off','Location','southwest')
%         legend('boxoff')
        history1=find(choices==1);
        history2=find(choices==2);
        scatter(history1,ones(1,length(history1))*12.5,'b','filled')
        scatter(history2,ones(1,length(history2))*12.5,'g','filled')
% 

            plot(1:SessionData.nTrials,weights(1,:),'--','Color','b','LineWidth',2.5);
            plot(1:SessionData.nTrials,weights(2,:),'--','Color','g','LineWidth',2.5);
        scatter(wcL,ones(1,length(wcL))*13,'b')
        scatter(wcR,ones(1,length(wcR))*13,'g')
        
        str=['Accuracy: ',num2str(acc), '       Alpha: ',num2str(alpha)];
        if whichModel=='SoftDec'
            str=['Accuracy: ',num2str(acc), '       Alpha: ',num2str(alpha), '       Beta: ' num2str(beta)];
        end
        if whichModel=='SoftMax'
            str=['Accuracy: ',num2str(acc), '       Alpha: ',num2str(alpha), '       Beta: ' num2str(beta)];
        end
        text(90,13.5,str)
    end
    
end