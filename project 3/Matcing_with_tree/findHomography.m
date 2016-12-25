% Input: matchedPoints1 - word (ref points) I x 2
%        matchedPoints2 - observed points I x 2
function [H, maxInliers] = findHomography(matchedPoints1, matchedPoints2)
    
    H = eye(3, 3);
    
    I = size(matchedPoints1, 1);
    maxInliers = 0;
    
    if I >= 4;
    
        SubsetPoints = 4;
        % random chose 4 distinct matched points
        

        threshold = 1;

        for t = 1:100
            % random chose 4 distinct matched points
            chose = randperm(I, SubsetPoints);

            % and the remain matched points are use to validate the homography
            otherMatchedPoints1 = matchedPoints1;
            otherMatchedPoints1(chose, :) = [];
            otherMatchedPoints2 = matchedPoints2;
            otherMatchedPoints2(chose, :) = [];

            tempH = estimateHomography(matchedPoints1(chose, :), matchedPoints2(chose, :));

            newMatchedPoints1 = tempH*[otherMatchedPoints1';ones(1, I - SubsetPoints)];
            newMatchedPoints1(1, :) = newMatchedPoints1(1, :) ./ newMatchedPoints1(3, :);
            newMatchedPoints1(2, :) = newMatchedPoints1(2, :) ./ newMatchedPoints1(3, :);
            newMatchedPoints1(3, :) = [];

            diff = sqrt((newMatchedPoints1(1,:) - otherMatchedPoints2(:, 1)').^2 ...
                + (newMatchedPoints1(2,:) - otherMatchedPoints2(:, 2)').^2);

            inliers = size(diff(diff < threshold), 2);
            if inliers > maxInliers
                maxInliers = inliers;
                H = tempH;
            end
        end
    end
end