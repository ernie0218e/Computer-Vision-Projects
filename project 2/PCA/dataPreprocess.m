% Filename: dataPreprocess.m
% Purpose: Preprocess data and save them in the form of .mat

% open and read the original data
load('data.mat');

% set up some constant parameters
imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

% get the training data
data = (train_data(1:imageNum, 2:size(train_data, 2)));
label = train_data(1:imageNum, 1);

% get the testing data
f_test_size = size(test_data, 1);
f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
f_test_label = test_data(1:f_test_size, 1);

disp('End of read Data');

% produce 14 kinds of different PCAs
% new dimension of data: {50, 100, 150,..., 700}
for i = 1:14
    
    % file name
    d = 50*i;
    saveFile = strcat('data_PCA_', int2str(d), '.mat');
    
    % compute PCA
    [data_t, V] = pca(data, d);
    % transform test data via V
    f_test_data_t = (V' * f_test_data')';
    
    % standardize all data
    data_t = standardize(data_t);
    f_test_data_t = standardize(f_test_data_t);
    
    % regroup data and label
    train_data = [label data_t];
    test_data = [f_test_label f_test_data_t];
    
    % save all data in .mat file
    save(saveFile, 'train_data', 'test_data');
    
    display(strcat('End of count ', int2str(i)));
end

saveFile = 'data_std.mat';

% standardize original data
data_t = standardize(data);
f_test_data_t = standardize(f_test_data);

% regroup data and label
train_data = [label data_t];
test_data = [f_test_label f_test_data_t];

% save all data in .mat file
save(saveFile, 'train_data', 'test_data');