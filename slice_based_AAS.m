% Dalton H Bermudez
% Marshall Foundation (Marshall Plan Scholarship)
% July 15, 2017
%
%
% Objective - to re-implement the methodology of Gradient Artifact
% subtraction technic from:"A Method for Removing Imaging Artifact 
% from Continuous EEG Recorded during Functional MRI"
% Philip J. Allen
% NeuroImage 12, 230?239 (2000)
% and
% Reference layer adaptive  ltering (RLAF) for EEG artifact 
% reduction in simultaneous EEG-fMRI
% David Steyrl, Gunther Krausz, Karl Koschutnig, 
% Günter Edlinger and Gernot R Müller-Putz
% J. Neural Eng. 14 (2017) 026003 (20pp)

clc
close all
clear all 
% load data from .mat file of simultaneous EEG data
EEG_contaminated = load('eeg_fmri_bi2_000005_Raw_Data.mat');
C = struct2cell(EEG_contaminated);
Marker_Count = EEG_contaminated.MarkerCount;
% read in markers the represent when was a volume acquired (TR)
marker = EEG_contaminated.Markers;

% Find positions in the EEG signal recording where a whole
% FMRI volume was acquired (interval of determined by the 
% repetition time)
for i = 1:length(marker)
    if(strcmp(marker(:,i).Description,'T  1_on') == 1)
        Marker_slic_on{i} = marker(:,i).Position;
    elseif(strcmp(marker(:,i).Description, 'T  1_off') == 1)
        Marker_slic_off{i} = marker(:,i).Position;
    end
end

Trig_on = cell2mat(Marker_slic_on);
Trig_off = cell2mat(Marker_slic_off);
Trig = [Trig_on, Trig_off];

%order the trigger marker in ascending order
Trig_order = sort(Trig);
Fs = EEG_contaminated.SampleRate;
nyq = Fs/2;



L = 5;% interpolate at least so the sample frequency is 20kHz
Y = 0;
K = 0;
% loop through each channels in the simultaneous EEG
for p = 1:EEG_contaminated.ChannelCount
    
    W = 0;
    for g = 1:2:length(Trig_order)-1
            W = W+1;
            % extract intervals in channel where FMRI volumes were acquire
            trial_EEG(:,W) = C{p}(Trig(:,g):Trig(:,g+1),:);
    end


    t_end = length(C)*1/Fs;

    t = (1:t_end*Fs)/Fs;%compute a time vector 

    drift = mean(trial_EEG,1);
    C_mat_tmp = trial_EEG - drift;% subtract the mean for every segment
                    % were a volume was acquire
    lbl = zeros([1,size(C_mat_tmp,2)]);
    num_epoch = size(C_mat_tmp,2);

    for j = 1:num_epoch
        I_C_matrix(:,j) = interp(C_mat_tmp(:,j),L,4,1);% interpolate each
                    % of the segment containing the GA
            
            % shift all the segments where a volume was acquire with
            % respect the segment of the first volume acquisition, until
            % the cross-correlation between the diffeerent segments are
            % maximatly correlated.
            [acor, lag] = xcorr(I_C_matrix(:,1), I_C_matrix(:,j));
            [mcor, pst] = max(abs(acor));
            diff_lag= lag(pst); 

         I_C_matrix(:,j) = circshift(I_C_matrix(:,j), diff_lag); 

        if j == 1
            lbl(j) = 2;
        elseif (2<= j) && (j<= 51)
                lbl(j) = 1;
        elseif (num_epoch - 49<= j) && (j<= num_epoch)
                lbl(j) = 3;
        end
    end
    % lbl = circshift(lbl,num_epoch/2); 
    local_more = find(lbl == 1);
    local_less = find(lbl == 3);
    ref_epoc = find(lbl == 2);
    indx_l = length(local_less);
    indx_m = length(local_more);
    I_C_matrix_tmp = I_C_matrix;
    I_C_M_N = zeros(size(I_C_matrix));
    
    
    index = zeros([1, EEG_contaminated.ChannelCount]);
    
    if(strcmp(EEG_contaminated.Channels(p).Name,'C3') == 1 || ...
            strcmp(EEG_contaminated.Channels(p).Name,'Cz') == 1 ...
            || strcmp(EEG_contaminated.Channels(p).Name,'22')==1 || ...
            strcmp(EEG_contaminated.Channels(p).Name,'21') ==1)
        K = K + 1;
     index(:,K) = 1; % variable that makers channel number of the 4 channels of interest
    end
    % loop through every epoch 
    
    for i = 1:num_epoch
        % for every iteration, calculate the average artifact template by
        % averaging 50 epochs before and 50 epochs after a given epoch
        join_matrix_1 = mean(I_C_matrix_tmp(:,local_less(1):local_less(indx_l)),2);
        join_matrix_2 = mean(I_C_matrix_tmp(:,local_more(1):local_more(indx_m)),2);
        
        Aver_templ = (join_matrix_1+join_matrix_2)./2;
        % reduce the mean square erro between the the calculated average
        % template and the given epoch
        alpha = inv(Aver_templ'*Aver_templ)*Aver_templ'*I_C_matrix_tmp(:,i);
        %shift the average template until maximatly correlated with the 
        % epochs is going to be subtracted from.
        [acor, lag] = xcorr(I_C_matrix(:,i), alpha*Aver_templ);
        [mcor, pst] = max(abs(acor));
        diff_lag= lag(pst); 

        Aver_templ = circshift(alpha*Aver_templ, diff_lag); 

        % Subtract the average template from the epoch 
        I_C_M_N(:,i) = I_C_matrix(:,i) - Aver_templ;

        % shift the label markers to do the same steps for 
        % consecutive epochs
        lbl = circshift(lbl,1);
        local_more = find(lbl == 1);
        local_less = find(lbl == 3);
        
    end
    
     EEG_tmp_1 = zeros(size(I_C_M_N,1)/L,size(I_C_M_N,2));

    
    for o=1:size(I_C_matrix,2)
        % down-smaple the interpolates signal to it orginal
        % sampling rate
        EEG_tmp_1(:,o) = decimate(I_C_M_N(:,o),L);
        
    end
    EEG_AAS = EEG_tmp_1(:);
    % low-pass filter at 70 Hz
    hpf = 70;
    Fs_EEG = 250; % in Hertz
    filtorder=round(1.2*Fs*L/(hpf-10));
    f=[0 (hpf-10)/(nyq*L) (hpf+10)/(nyq*L) 1]; 
    a=[0 0 1 1];
    hpfwts=firls(filtorder,f,a);
    r = filtfilt(hpfwts,1, EEG_AAS);
    E_A = EEG_AAS  - r;
    EEG_C3 = C_mat_tmp(:);
    
    % Down-sample signal again to a sampling frequency 
    % of 250 Hz
    E{p} = decimate(E_A, Fs/Fs_EEG);
    
end
% save the new processed EEG signal after performing 
% Average Artifact Subtraction for the Gradient artifact
clear EEG_AAS_sub
EEG_GA_AAS = cell2mat(E);
S = C{35};
EEG_AAS_sub.selected_channels = S;
EEG_AAS_sub.classifaction_channel = index;
EEG_AAS_sub.data_sub = EEG_GA_AAS';
EEG_AAS_sub.Marker_pos = [marker(:).Position]/(Fs/Fs_EEG);
EEG_AAS_sub.sub_sampl_freq = Fs_EEG;
EEG_AAS_sub.Marker_descrition = [marker(:).Description];

save('EEG_AAS.mat', '-struct', 'EEG_AAS_sub')

