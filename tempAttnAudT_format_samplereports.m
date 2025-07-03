%% create_FIRDecon_datasets
% Using the sample reports from the Eyelink Viewer, create .csv files that
% will be formatted for the pupil preprocessing toolbox in Python [based on
% Knapen et al. 2017
% The sample reports from the Viewer are named e.g. "s1_run3.txt"

clear all;clc
for s = 7:27
    for r = 1:4
        sub = num2str(s);
        run = num2str(r);
        %cd Sample_reports
        tdfread(['s',sub,'_run',run,'.txt'])
        
        %% SET UP DATASET LABELS
        filenamePupil = ['FIR_s',sub,'_run',run,'_pupil.csv'];
        filenameTrial = ['FIR_s',sub,'_run',run,'_trials.csv'];
        filenameBlink = ['FIR_s',sub,'_run',run,'_blinks.csv'];
        filenameSaccs = ['FIR_s',sub,'_run',run,'_saccades.csv'];
        
        %% REFORMAT EYE DATA IF NECESSARY
        if length(RIGHT_PUPIL_SIZE(1,:)) == 7
            for i = 1:length(RIGHT_PUPIL_SIZE(:,1))
                if RIGHT_PUPIL_SIZE(i,1) == '.'
                    x(i,1) = 0;
                else
                    a = num2str(RIGHT_PUPIL_SIZE(i,1)); b = num2str(RIGHT_PUPIL_SIZE(i,2)); c = num2str(RIGHT_PUPIL_SIZE(i,3)); d = num2str(RIGHT_PUPIL_SIZE(i,4));
                    %e = num2str(RIGHT_PUPIL_SIZE(i,5)); f = num2str(RIGHT_PUPIL_SIZE(i,6)); g = num2str(RIGHT_PUPIL_SIZE(i,7));
                    x(i,1) = str2num([a b c d]);
                end
            end
        end
        
        if length(RIGHT_PUPIL_SIZE(1,:)) == 7
            clear RIGHT_PUPIL_SIZE a b c d e f g i
            RIGHT_PUPIL_SIZE = x;
            clear x ans
        end
        
        a(:,1) = TIMESTAMP; a(:,2) = RIGHT_PUPIL_SIZE;
        
        %% FIND SAMPLE MESSAGES
        count = 1; c_zero = 1; c_one = 1; c_two = 1; trialtype(1:144,1) = 0; typecount = 1;
        for i = 1:length(SAMPLE_MESSAGE)
            if strfind(SAMPLE_MESSAGE(i,:), 'condition.0.') > 1 % Find stimulus onsets that aren't scrambled images
                continue
            else
                if SAMPLE_MESSAGE(i) == 'v' % Onset of visual stimulus on an actual trial
                    trial_starts(count,1) = a(i,1); % timestamp
                    trial_starts(count,2) = a(i,2); % pupil size
                    trial_starts(count,3) = i;      % index
                    if strfind(SAMPLE_MESSAGE(i,:), 'toneType.0') > 1       % Condition 0
                        trial_starts(count,4) = 0;
                    elseif strfind(SAMPLE_MESSAGE(i,:), 'toneType.1') > 1   % Condition 1
                        trial_starts(count,4) = 1;
                    else                                                    % Condition 2
                        trial_starts(count,4) = 2;
                    end
                    count = count+1;
                end
            end
        end
        % trial_starts should now be 144 rows by 3 columns
        % tone_zero/one/two should now be 48x3
        trials(:,1) = trial_starts(:,1); trials(:,2) = trial_starts(:,4);
        
        %% INDEX THE BLINKS AND SACCADES
        b = 1; k = 1;
        for i = 2:length(RIGHT_PUPIL_SIZE)
            if RIGHT_IN_BLINK(i) ~= RIGHT_IN_BLINK(i-1) && RIGHT_IN_BLINK(i) == 1 % blink start
                blinks(b,1) = a(i,1); blink_index(b,1) = i;
            elseif RIGHT_IN_BLINK(i) ~= RIGHT_IN_BLINK(i-1) && RIGHT_IN_BLINK(i) == 0 % blink end
                blinks(b,2) = a(i,1); blink_index(b,2) = i-1; b = b+1;
            end
            if RIGHT_IN_SACCADE(i) ~= RIGHT_IN_SACCADE(i-1) && RIGHT_IN_SACCADE(i) == 1 % saccade start
                saccs(k,1) = a(i,1); sacc_index(k,1) = i;
            elseif RIGHT_IN_SACCADE(i) ~= RIGHT_IN_SACCADE(i-1) && RIGHT_IN_SACCADE(i) == 0 % saccade end
                saccs(k,2) = a(i,1); sacc_index(k,2) = i-1; k = k+1;
            end
        end
        
        if blinks(length(blinks(:,1)),2) == 0; blinks(length(blinks(:,1)),2) = a(length(a(:,1)),1); end
        if saccs(length(saccs(:,1)),2) == 0; saccs(length(saccs(:,1)),2) = a(length(a(:,1)),1); end
        
        %%  MAKE PUPIL DATASET
        fid = fopen(filenamePupil, 'wt');
        fprintf(fid, 'timepoints,pupil\n');  % header
        dlmwrite(filenamePupil, a, 'precision', 10, '-append')
        fclose(fid);
        
        %% MAKE TRIAL DATASET
        fid = fopen(filenameTrial, 'wt');
        fprintf(fid, 'trial_start, tonetype\n');  % header
        dlmwrite(filenameTrial, trials, 'precision', 10, '-append')
        fclose(fid);
        
        %% MAKE BLINK AND SACCADE DATASETS
        fid = fopen(filenameBlink, 'wt');
        fprintf(fid, 'blink_start,blink_end\n');  % header
        dlmwrite(filenameBlink, blinks, 'precision', 10, '-append')
        fclose(fid);
        
        fid = fopen(filenameSaccs, 'wt');
        fprintf(fid, 'saccs_start,saccs_end\n');  % header
        dlmwrite(filenameSaccs, saccs, 'precision', 10, '-append')
        fclose(fid);
        
        clearvars -except s r
    end
end