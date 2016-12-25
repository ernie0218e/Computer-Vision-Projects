% Filename: estimateHomography.m
% Purpose: estimage homography using Gauss-Newton method
% Input: matchedPoints1 - word (ref points) [each row (u, v)] I x 2
%        matchedPoints2 - observed points [each row (x, y)] I x 2
function [H] = estimateHomography(matchedPoints1, matchedPoints2)
    
    I = size(matchedPoints1, 1);
    
    A = zeros(2*I, 9);
    for i = 1:I
        A(2*i - 1, 4:6) = [-matchedPoints1(i, :) -1];
        A(2*i - 1, 7) = matchedPoints1(i, 1)*matchedPoints2(i, 2);
        A(2*i - 1, 8) = matchedPoints1(i, 2)*matchedPoints2(i, 2);
        A(2*i - 1, 9) = matchedPoints2(i, 2) - matchedPoints1(i, 2);
        A(2*i, 1:3) = [matchedPoints1(i, :) 1];
        A(2*i, 7) = -matchedPoints1(i, 1)*matchedPoints2(i, 1);
        A(2*i, 8) = -matchedPoints1(i, 2)*matchedPoints2(i, 1);
        A(2*i, 9) = matchedPoints1(i, 1) - matchedPoints2(i, 1);
    end
    
    % compute svd of A
    [U, L, V] = svd(A);
    
    % use last column of V as initial value
    phi = [V(1:8, 9)./V(9,9); 0];
    
    lambda = 0.001;
    
    threshold = 1e-5;
    
    % start of Gauss Newton method
    while true
        
        % reshape phi into a 3 x 3 matrix
        phiMat = eye(3, 3);
        tempPhiMat = reshape(phi, 3, 3);
        phiMat = phiMat + tempPhiMat';

        x_telta = phiMat*[matchedPoints1(:, 1)';matchedPoints1(:, 2)';ones(1, I)];
        
        % compute the estimated point
        x_telta(1, :) = x_telta(1, :) ./ x_telta(3, :);
        x_telta(2, :) = x_telta(2, :) ./ x_telta(3, :);

        % form the error vector
        e = [(matchedPoints2(:, 1)' - x_telta(1, :));...
            (matchedPoints2(:, 2)' - x_telta(2, :))];

        e = reshape(e, 2*I, 1);
        
        % compute Jacobian matrix
        % combine all data into the same matrix
        J = zeros(2*I, 8);
        for i = 1:I
            J(2*i - 1, 1:3) = [matchedPoints1(i, :) 1];
            J(2*i - 1, 7) = -matchedPoints1(i, 1)*x_telta(1, i);
            J(2*i - 1, 8) = -matchedPoints1(i, 2)*x_telta(1, i);
            J(2*i - 1, :) = J(2*i - 1, :) ./ ...
                (matchedPoints1(i, 1)*phiMat(3,1)+matchedPoints1(i, 2)*phiMat(3,2)+1);
            J(2*i, 4:6) = [matchedPoints1(i, :) 1];
            J(2*i, 7) = -matchedPoints1(i, 1)*x_telta(2, i);
            J(2*i, 8) = -matchedPoints1(i, 2)*x_telta(2, i);
            J(2*i, :) = J(2*i, :) ./ ...
                (matchedPoints1(i, 1)*phiMat(3,1)+matchedPoints1(i, 2)*phiMat(3,2)+1);
        end

        A = J'*J;
        
        % to lower the condition of A
        maxVal = max(max(A));
        % normalize A
        A = A ./ maxVal;
        
        % the b should be normalized by the same factor
        b = J'*e ./ maxVal;
        
        % if cond(A + lambda*eye(8, 8)) is too large
        if cond(A + lambda*eye(8, 8)) > 1e10
            H = eye(3, 3);
            return;
        end
        
        delta_phi = (A + lambda*eye(8, 8))\b;
        
        % update phi
        phi(1:8, 1) = phi(1:8, 1) + delta_phi;
        
        if sum(delta_phi.^2, 1) < threshold
            break;
        end
            
    end
    
    H = eye(3, 3);
    tempH = reshape(phi, 3, 3);
    H = H + tempH';
end