function [e] = testNonlinearMulticlassLogistic(data, label, phi_zero, phi, zeta, K)
    
    I = size(data, 1);
    error = zeros(10, 1);

    % act = phi_zero
    a = repmat(phi_zero, 1, I);
    % act = \phi_0+\sum_{k=1}^{K}{\phi_katan(\zeta^{T}\mathbf{X})}
    for k = 1:K
            a = a + repmat(phi(:, k), 1, I).*atan(squeeze(zeta(:, :, k))*data');
    end
    % softmax
    den = sum(exp(a), 1);
    lambda = exp(a) ./ repmat(den, 10, 1);

    for i = 1:I

        [maxVal, maxLabel] = max(lambda(:, i));
        maxLabel = maxLabel - 1;

        if maxLabel ~= label(i)
            error(label(i) + 1) = error(label(i) + 1) + 1;
        end
    end

    e = sum(error)/I;

end