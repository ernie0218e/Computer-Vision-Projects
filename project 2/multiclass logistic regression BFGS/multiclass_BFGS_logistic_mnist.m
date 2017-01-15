clear;

% open and read the data
load('data_PCA_50.mat');


imageNum = 60000;
ratio = 1;
imageSize = round((sqrt(784)*ratio)^2);
imageWidth = round(sqrt(imageSize));

% get the training data
data = (train_data(1:imageNum, 2:size(train_data, 2)));
label = train_data(1:imageNum, 1);

% get the testing data
f_test_size = size(test_data, 1);
f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
f_test_label = test_data(1:f_test_size, 1);

disp('End of read Data');

% data = data./255;
% f_test_data = f_test_data./255;

% [data, V] = pca(data, 200);
% 
% f_test_data = (V' * f_test_data')';

disp('End of process image');


data = [ones(imageNum, 1) data];

f_test_data = [ones(f_test_size, 1) f_test_data];


classNum = 10;
ita = 0.01;
w = label + 1;
x = data;

[phi] = multiLogisticBFGS(w, x, classNum, ita);

error = zeros(10, 1);
[lambda] = linearSoftMax(phi, f_test_data);
for i = 1:f_test_size
    
    [maxVal, maxLabel] = max(lambda(:, i));
    maxLabel = maxLabel - 1;
    
    if maxLabel ~= f_test_label(i)
        error(f_test_label(i) + 1) = error(f_test_label(i) + 1) + 1;
    end
end

e = sum(error)/f_test_size;


train_error = zeros(10, 1);
[lambda] = linearSoftMax(phi, data);
for i = 1:imageNum
    
    [maxVal, maxLabel] = max(lambda(:, i));
    maxLabel = maxLabel - 1;
    
    if maxLabel ~= label(i)
        train_error(label(i) + 1) = train_error(label(i) + 1) + 1;
    end
end

train_e = sum(train_error)/imageNum;