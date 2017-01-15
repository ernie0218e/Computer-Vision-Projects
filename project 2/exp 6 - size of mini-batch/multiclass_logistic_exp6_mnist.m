clear;

imageNum = 60000;
imageSize = 784;
imageWidth = round(sqrt(imageSize));

loadFileName = 'data_PCA_200.mat';
    
% open and read the data
load(loadFileName);

segs = [1;10;100;1000;10000];

error = zeros(500, size(segs, 1));
tr_error = zeros(500, size(segs, 1));

for i = 1:size(segs, 1)

    % get the training data
    data = (train_data(1:imageNum, 2:size(train_data, 2)));
    label = train_data(1:imageNum, 1);

    % get the testing data
    f_test_size = size(test_data, 1);
    f_test_data = (test_data(1:f_test_size, 2:size(test_data, 2)));
    f_test_label = test_data(1:f_test_size, 1);
    disp('End of read Data');


    % train 1 hidden layer NN model
    classNum = 10;
    eta = 0.5;
    alpha = 0.2;
    seg = segs(i);
    w = label + 1;
    x = data;

    [W_xh, W_hy, theta_h, theta_y, error(:, i), tr_error(:, i)]...
        = MLP(x, w, classNum,...
            f_test_data, f_test_label, eta, alpha, seg);

end


figure(1);
for i = 1:size(error, 2)
    plot(error(error(:,i) > 0,i));
    hold on;
    plot(tr_error(tr_error(:,i) > 0,i));
    hold on;
end
title('Training & Test error');
grid on;
xlabel('iterations');
ylabel('error rate');
legend('test-batch size=60000','train-batch size=60000',...
    'test-batch size=6000','train-batch size=6000',...
    'test-batch size=600','train-batch size=600',...
    'test-batch size=60','train-batch size=60',...
    'test-batch size=6','train-batch size=6');
        
% figure(2);
% for i = 1:size(tr_error, 2)
%     plot(tr_error(tr_error(:,i) > 0,i));
%     hold on;
% end
% title('Training error');
% grid on;
% xlabel('iterations');
% ylabel('error rate');
% legend('batch size=60000','batch size=6000',...
%     'batch size=600','batch size=60','batch size=6');