function [e] = testMLP(data, label, W_xh, W_hy, theta_h, theta_y)
    
    I = size(data, 1);
    error = zeros(10, 1);

    Y = MLP_Predict(data, W_xh, W_hy, theta_h, theta_y);

    for i = 1:I
        [maxVal, maxLabel] = max(Y(:, i));
        maxLabel = maxLabel - 1;

        if maxLabel ~= label(i)
            error(label(i) + 1) = error(label(i) + 1) + 1;
        end
    end

    e = sum(error)/I;
end