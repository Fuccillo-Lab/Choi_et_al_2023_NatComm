%% Example script for running Q model fitting and plotting

    if ~exist('SessionData','var')
       disp('load up a Bpod Session')
       uiopen 
    end
    
    %% Test Softmax
    softmaxResult=fitQModel_2CSR(SessionData,'SoftMax');
    plot2CSR(softmaxResult);
    %% Test Softmax Decay
    softmaxDecayResult=fitQModel_2CSR(SessionData,'SoftDec');
    plot2CSR(softmaxDecayResult);
    %% Test Epsilon
    epsilonResult=fitQModel_2CSR(SessionData,'Epsilon');
    plot2CSR(epsilonResult);
    %% Test Epsilon Decay
    epsilonDecayResult=fitQModel_2CSR(SessionData,'EpsiDec');
    plot2CSR(epsilonDecayResult);
    %% Testing
    clear('SessionData'); %%SessionData is now saved in each variable