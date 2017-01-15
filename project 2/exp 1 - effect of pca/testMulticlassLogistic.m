function [e] = testMulticlassLogistic(data, label, phi)
    
    I = size(data, 1);
    error = zeros(10, 1);
    
    [lambda] = linearSoftMax(phi, data);
    for i = 1:I

        [maxVal, maxLabel] = max(lambda(:, i));
        maxLabel = maxLabel - 1;

        if maxLabel ~= label(i)
            error(label(i) + 1) = error(label(i) + 1) + 1;
        end
    end

    e = sum(error)/I;
end