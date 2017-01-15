% Filename: charSegmentation.m
% Purpose: segment and extract characters inside the license plate
% Input: plate - input image of license plate with
%                white background (R x C)
%        numOfChar - number of candidate characters
% Output: char - output binary image of chararacters
%
function [charImg] = charSegmentation(plate, numOfChar)
    
    % define the threshold of 'white' area
    w_ratio_thres = 0.15;
    
    % define the minimum length 
    % that can be called 'region'('white line')
    charThres = 3;
    
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
    
    % extract the license plate
    plate = plate(v_lowerBound:v_upperBound, :);
    
    % define the minimum number of 'white line'
    numOfBoundaryMin = numOfChar - 1;
    
    % histogram of horizontal projection
    h_proj = sum(plate, 1);

    % find the threshold that can extract all characters
    h_thres = max(h_proj);
    h_peaks = h_proj > h_thres;
    while true
        
        regions = 0;
        first_pt = 1;
        last_pt = 2;
        while last_pt <= length(h_peaks)
            if h_peaks(first_pt) ~= h_peaks(last_pt)
                if h_peaks(first_pt) == 1
                    regions = regions + 1;
                end
                first_pt = last_pt;
            end
            last_pt = last_pt + 1;
        end
        if h_peaks(last_pt - 1) == 1
           regions = regions + 1; 
        end
        
        if numOfBoundaryMin <= regions
            break;
        elseif h_thres <= 1
            break;
        end

        h_thres = h_thres - 1;
        h_peaks = h_proj > h_thres;
        
    end
    
    if h_thres ~= 1

        % get the character
        first_pt = 1;
        last_pt = 2;
        idx = 1;
        while last_pt <= length(h_peaks)
            if h_peaks(first_pt) ~= h_peaks(last_pt)
                if h_peaks(first_pt) == 0 && (last_pt-first_pt) > charThres
                    
                    % crop character
                    tempChar = 1 - plate(:, first_pt:(last_pt - 1));
                    
                    % calculate ratio of white area to whole area
                    w_ratio = sum(sum(tempChar))...
                               /(size(tempChar, 1)*size(tempChar, 2));
                    % the ratio is greater than the threshold
                    if w_ratio >= w_ratio_thres
                        % this image can be a charater
                        charImg{idx, 1} = tempChar;
                        idx = idx + 1;
                    end
                end
                first_pt = last_pt;
            end
            last_pt = last_pt + 1;
        end
        if h_peaks(first_pt) == 0 && (last_pt-first_pt) > charThres
            % crop character
            tempChar = 1 - plate(:, first_pt:(last_pt - 1));

            % calculate ratio of white area to whole area
            w_ratio = sum(sum(tempChar))...
                       /(size(tempChar, 1)*size(tempChar, 2));
            if w_ratio >= w_ratio_thres
                charImg{idx, 1} = tempChar;
            end
        end
    else
        charImg = cell(1, 0);
        charImg{1} = zeros(1, 1);
    end
    
end