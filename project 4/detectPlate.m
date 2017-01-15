% 
% Filename: detectPlate.m
% Purpose: check whether the texture of given image is matched with the one
%          of license plate
% Input: plate - the image of the candidate license plate
%        numOfChar - number of character inside the license plate
%        lparMax - maximum aspect ratio of license plate
%        lparMin - minimum aspect ratio of license plate
% Output: isPlate - indicate whether the image is the license plate
%         regions - number of 'white' gaps inside the image
function [isPlate, regions] = detectPlate(plate, numOfChar, lparMax, lparMin)
    
    % binarize the image
    plate = imbinarize(plate);
    
    % histogram of vertical projection
    v_proj = sum(plate, 2);
    
    % define the threshold of 'white line'
    v_thres = max(v_proj)*0.9;
    
    % get the white line
    v_peaks = v_proj > v_thres;
    
    % find the boundary of top side and bottom side
    v_lowerBound = 1;
    v_upperBound = length(v_peaks);
    half_pt = round(length(v_peaks) / 2);
    for i = 1:half_pt
        if v_peaks(i) == 1
            v_lowerBound = i;
        end
    end
    for i = length(v_peaks):-1:(length(v_peaks) - half_pt + 1)
        if v_peaks(i) == 1
            v_upperBound = i;
        end
    end
    
    % extract license plate
    plate = plate(v_lowerBound:v_upperBound, :);
    
    % check the aspect ratio of extracted image
    lpar = size(plate, 2)/size(plate, 1);
    
    if lpar >= lparMin && lpar <= lparMax
    
        % if the image still sticks to the rule
        
        % find the horizonatl projection
        h_proj = sum(plate, 1);
        
        % define the threshold of 'white line'
        h_thres = max(h_proj)*0.9;
        
        % define the minimum length 
        % that can be called 'region'('white line')
        charThres = 3;

        % get the region which is greater than the threshold
        h_peaks = h_proj > h_thres;

        % calculate the number of the 'white line'
        regions = 0;
        first_pt = 1;
        last_pt = 2;
        while last_pt <= length(h_peaks)
            if h_peaks(first_pt) ~= h_peaks(last_pt)
                if h_peaks(first_pt) == 1 && (last_pt-first_pt) > charThres
                    regions = regions + 1;
                end
                first_pt = last_pt;
            end
            last_pt = last_pt + 1;
        end
        if h_peaks(last_pt - 1) == 1
           regions = regions + 1; 
        end
    
        % if the number of the region (# of white lines) is similar
        % to the one inside the license plate
        if regions >= numOfChar-1 && regions <= numOfChar+1
            isPlate = true;
        else
            isPlate = false;
        end
    else
        isPlate = false;
        regions = 0;
    end
end