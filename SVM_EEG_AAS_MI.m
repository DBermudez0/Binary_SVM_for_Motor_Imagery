% Dalton H Bermudez
% Marshall Foundation (Marshall Plan Scholarship)
% August 16, 2017
%
%
% SVM - Training and Classification of R hand and R feet Motor Imagery 
% Objective: Compare the accuracy of classification of L and R Motor
% Imagery to its true label compare to the accuracy of classification of
% R-hand and R-feet Motor Imagery data of EEG (after GA removal and enhancement of MI
% activity through GMM-EM) inside FMRI scanner.
% 
%
% Assumption: Since the EEG was samples at a sampling frequency of 200Hz,
% the highest frequecy represented in the signal is 100 Hz (Fs/2). Therefore,
% since 100Hz  is lower than 1000 Hz, the inductive effects and
% electromagnetic propagation can be neglected. Therefore, it can be
% assumed that the electrical signal recorded from the electrodes on top
% the Parietal Lobe mainly detect brain waves from the activations of
% neurons in the Parietal cortex.
%
% Only Used EEG Channels located in Parietal Lobe
% Channels used:
% C3 - left hemisphere (right hand MI)
% Cz - right feet MI
% C3 - left hemisphere (right hand MI)
% Cz - right feet MI
% C22 - right feet MI
% C21 - left hemisphere (right hand MI)

clc
close all
clear all

% parameters for EEG pre-processing through: Band-Pass Filtering
% for band pass of Mu brain wave
low_freq_2 = 8; % in Hertz
high_freq_2 = 30; % in Hertz

low_notch = 49.5;
high_notch = 50.5;


ftype_2 = 'bandpassfir';

load EEG_AAS
Fs = sub_sampl_freq;
% Design_1 = designfilt(ftype_2,'FilterOrder', 1000,'CutoffFrequency1', low_freq_2,...
%             'CutoffFrequency2', high_freq_2,'SampleRate', Fs);
Data = data_sub';

% read in the data from the paradigm text file
fid_1 = fopen('1out.txt');
C = textscan(fid_1, '%f %s %f %s %f', 'delimiter',',');
fclose(fid_1);
% read in the time, whether the fixated cross was on or off,
% the number of trials, and the type of MI performed
f = {'Time','Description','Trial_num', 'Type', 'Type_MI'};
s_paradigm = cell2struct(C,f,2);

num_Descrip = length(s_paradigm.Description);

u = 0;
a = 0;
% Determine the location in the EEG signal (sample position) 
% where the fixated cross was on and when it switched off
for i = 1:num_Descrip
    
    if(strcmp(s_paradigm.Description{i}, 'new trial') == 1)
        u =  u +1;
        pos_stim{u} = i; 
    elseif(strcmp(s_paradigm.Description{i}, 'cross off') == 1)
        a = a + 1;
        pos_stim_off{a} = i;
    end
    
end

pos_stim = cell2mat(pos_stim);
% Extract the time intervals were the participant performend
% the Motor Imagery
time_stim = s_paradigm.Time(pos_stim,:);
labels_orig = s_paradigm.Type_MI(pos_stim,:);

% read the Band-passed (1-80Hz) and notch filtered at 50 Hz
for j = 1:7
    S =  load(['Data_prepross' num2str(j) '.mat']);
    X{j} = S.Data_pre(S.Channel_pos_fil(1):S.Channel_pos_fil(2)- ...
        S.Channel_pos_fil(1):S.Channel_pos_fil(2),:,:);%%%
    L{j} = S.orig_clean_EEG_lbl;
    Whol{j} = S.EEG_bandpass;
end
% FIR band-pass filter at range of mu and beta requency ranges (8-30 Hz)
Design = designfilt(ftype_2,'FilterOrder', 1000,'CutoffFrequency1', low_freq_2,...
            'CutoffFrequency2', high_freq_2,'SampleRate', S.Samplef);
% resample the simultaneous EEG signals of sampling frequency of 250Hz
% to the sampling frequency of the non-simultaneous EEG of 200 Hz
for C = 1:size(Data, 2)
    y(:,C) = resample(double(Data(:,C)),S.Samplef,Fs);
end

% time interval were MI was performed
time_in_data = time_stim(time_stim <= size(y,1)/S.Samplef - .1);
labels_orig = labels_orig(1:length(time_in_data));

% location in samples were the fixated cross appeared on the screen 
% to indicate the participant to perform MI
cross_on_samp = round(time_in_data*S.Samplef);

duration_for_learn = .1; % 100ms after stimulus
u = 0;

Y = zeros(size(cross_on_samp,1),duration_for_learn*S.Samplef, size(y, 2));

% Extract the segments of the EEG signal where MI was perform
for T = 1:size(labels_orig, 1)
  Y(T,:,:) = y(cross_on_samp(T,:):(cross_on_samp(T,:)+duration_for_learn*S.Samplef-1),:);
end

Y_1 = Y(:,:,1:2);
% reshape segments of MI performed into a feature vector
Y = reshape(Y_1, size(Y_1,1), size(Y_1,2)*size(Y_1,3));

X_clean_EEG = cell2mat(permute(X, [1, 3, 2]));
% reshape the dimensionality of the non-simultaneous EEG, so an FIR
% bandpass filter can be perormed on the from 8-30Hz
X_trials = reshape(X_clean_EEG, size(X_clean_EEG, 1), size(X_clean_EEG, 2)*size(X_clean_EEG, 3));
X_bndPass = single(filtfilt(Design,double(X_trials')));
X_clean_EEG = reshape(X_bndPass, size(X_clean_EEG, 1), size(X_clean_EEG, 2),size(X_clean_EEG, 3));
Lbl_clean_EEG = cell2mat(L');
Lbl_clean_EEG = Lbl_clean_EEG';

duration_for_learn = .1; % 100ms after stimulus
cl_X = permute(X_clean_EEG, [3, 2, 1]);

% reshape the segments were MI was performend in the non-simultaneous EEG 
% needed to learn a prior of how MI brain activity should look like 
% (The extracted segments are 100 ms seconds long)
cl_X_1 = cl_X(:,:,1:2);% only choose C3 and Cz since are most representative
    % channels for detecting right hanf MI and right feet MI, respectively.
cl_X_pre = cl_X_1(:,1:(duration_for_learn*S.Samplef),:);
cl_X = reshape(cl_X_pre, size(cl_X_pre,1), size(cl_X_pre,2)*size(cl_X_pre,3));

% check for assumption of independence
MI_1_only = cl_X(find(Lbl_clean_EEG == 1), :);
MI_2_only = cl_X(find(Lbl_clean_EEG == 2), :);
MI_s = [MI_1_only; MI_2_only];
MI_s = circshift(MI_s, size(MI_2_only,1)/3);
Lbl = [ones(size(cl_X,1)/2, 1); zeros(size(cl_X,1)/2, 1)];
Lbl(Lbl == 0) = 2;
Lbl_shift = circshift(Lbl, size(MI_2_only,1)/3);
Covar_MI = cov(MI_1_only, MI_2_only);

% plot to see if there are gaussian distrubuted with zero mean:
gscatter(MI_s(1:105,:), MI_s(end:-1:106,:), Lbl_shift(1:105))
xlabel('mean of trial feature vector for right hand and feet motor imagery')
ylabel('mean of trial feature vector for right hand and feet motor imagery')

permutations = 2;
k = 2; % number of mixture components (one for each type of MI)-right hand or rigth feet
Q = zeros(size(Data));
% interate over the C3 and Cz channels segment of (100ms Long) MI
% performance
for O = 1:size(Data,2)
    for g = 1:permutations
        % randomly select 100ms sections of MI performance from either C3
        % and Cz channels from non-simultaneous EEG
        indx = randperm(size(cl_X,1), size(cl_X, 1)/permutations);
         % maximum iteration for convergance 1000
        options = statset('Display', 'iter', 'MaxIter', 1000); 
        % find parameters for GMM with 2 components
        obj = gmdistribution.fit(cl_X(indx,:),k, 'Options',options, 'CovType', ...
            'diagonal','Regularize',0.1);
        % extract the covariace matrixes for the right hand MI and
        % right feet MI compoenets
        ComponentCovariances = obj.Sigma;
        Cov_c_1 = ComponentCovariances(:,:,1);
        Cov_c_2 = ComponentCovariances(:,:,2);
        % find the general covarience matrix form the component
        % covarience matrixes 
        R_c_1 = mean(Cov_c_1(:));
        R_c_2 = mean(Cov_c_2(:));
        R_x_prior = R_c_1 + R_c_2;

        % implementation of Wiener filter with learned prior covarience 
        % matrix
        if g == 1
            Rx = cov(Data(:,O));
            Q_1(:,O) = (R_x_prior*inv(Rx))*y(:,O);
        else
             Q_1(:,O) = (R_x_prior*inv(Rx))*Q_1(:,O);
        end

        Rx = R_c_1 + R_c_2;
        
    end
end
nyq = S.Samplef/2;
% calculate the root mean square deviation for all channels
% in the Simultaneous EEG between the EEG signal after Gradient
% Artifact removal and EEG after both Gradient Artifact and
% Pulsatile Artifact removal.
err = y - Q_1;
RMSD = sqrt(mean(err.^2));

aver_RMSD = mean(RMSD(:,1:end-1));
table(aver_RMSD, 'VariableNames', {'Mean_RMSE'})
% notch filter 50 Hz power line and band pass 8-30 Hz
[b,a] = butter(4,[low_notch/nyq high_notch/nyq],'stop');
raw_Data_temp = single(filtfilt(b,a,double(y)));
filt_data_simul = single(filtfilt(Design,double(raw_Data_temp)));
size_trial = S.Samplef;

 for k = 1:size(filt_data_simul,2)
       for p = 1:length(labels_orig)
        trials_EEG_MI(k,:,p) = filt_data_simul(cross_on_samp(p,:):...
            cross_on_samp(p,:)+size_trial-1,k)';
       end
 end

permutations_1 = 4;
% choose the C3, Cz, C22, and C21 since there are located difectly above
% the patietal lobe and will be the ones that detect most of MI brain
% activity
trials_EEG_MI = trials_EEG_MI(logical(classifaction_channel),:,:);
X_train = trials_EEG_MI(1:2,:,:);
X_test = trials_EEG_MI(3:4,:,:);
% for power estimate feature use:
        % X_train= trials_EEG_MI(1:2,:,:).^2;
        % X_test = trials_EEG_MI(3:4,:,:).^2;
% reshape into time-domain features vectors
X_train_f = reshape(X_train, size(X_train,3), size(X_train,2)*size(X_train,1));
X_test_f = reshape(X_test, size(X_test,3), size(X_test,2)*size(X_test,1));

for E = 1:permutations_1
        indx_1 = randperm(size(X_train_f,1), size(X_train_f, 1)/permutations_1);
        svmStruct = fitcsvm(X_train_f(indx_1,:),labels_orig(indx_1,:),'KernelFunction', ...
            'linear','OutlierFraction', 0.05,'Standardize',true,'Verbose',1);
        CompactSVMModel = compact(svmStruct);
        
        [ScoreCSVMModel,ScoreParameters] = fitPosterior(CompactSVMModel,...
     X_test_f(indx_1,:),labels_orig(indx_1,:));
 
 
  [label{E},postProbs{E}] = predict(ScoreCSVMModel,X_test_f(indx_1,:));
  %mis-classification rate
 errRate = sum(labels_orig(indx_1,:) ~= label{E})/size(X_test_f(indx_1,:),1) 
         
 % display each of the testing set original labels,predicted labels and Posterior Probabilities
        table(labels_orig(indx_1,:), label{E},postProbs{E}(:,2),...
    'VariableNames',{'OriginalLabel','PredictedLabel','PosteriorProbability'})  

     gd_portProbs = find(postProbs{E}(:,2) > 0.9);
     best_indx{E} = indx_1(:,gd_portProbs);
     
end