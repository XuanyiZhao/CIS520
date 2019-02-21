tic
% Load training data and test data
X_train_full = load('../Data-set-2/X_train.txt');
y_train_full = load('../Data-set-2/y_train.txt');
X_test_full = load('../Data-set-2/X_test.txt');
y_test_full = load('../Data-set-2/y_test.txt');

X_train(:,:,1) = [load(['../Data-set-2/CrossValidation/Fold1/X_100%.txt']), ones(145,1)];
Y_train(:,:,1) = load(['../Data-set-2/CrossValidation/Fold1/y_100%.txt']);

for i = 2:5
    X_train(:,:,i) = [[load(['../Data-set-2/CrossValidation/Fold',num2str(i),'/X_100%.txt']); zeros(1,8)], ones(145,1)];
    Y_train(:,:,i) = [load(['../Data-set-2/CrossValidation/Fold',num2str(i),'/y_100%.txt']); zeros(1,1)];
end

X_train(:,end,:) = [];

initparams = [7, 10, 15, 17, 20];
init_b1 = {};
init_b2 = {};
init_w1 = {};
init_w2 = {};
for p = 1:length(initparams)
    init_b1{p} = load(['../Data-set-2/InitParams/',num2str(initparams(p)),'/b1.txt']);
    init_b2{p} = load(['../Data-set-2/InitParams/',num2str(initparams(p)),'/b2.txt']);
    init_w1{p} = load(['../Data-set-2/InitParams/',num2str(initparams(p)),'/W1.txt']);
    init_w2{p} = load(['../Data-set-2/InitParams/',num2str(initparams(p)),'/W2.txt']);
end

training_error = [];
test_error = [];
cross_validation_error = [];
for j = 1:5
    average_error = [];
    for k = 1:5
        temp_x = X_train;
        temp_y = Y_train;
        if (sum(temp_x(end,:,k)) ~= 1) && (temp_y(end,:,k) ~= 0)
            X_test = temp_x(:,:,k);
            Y_test = temp_y(:,:,k);
        elseif (sum(temp_x(end,:,k)) == 1) || (temp_y(end,:,k) == 0)
            X_test = temp_x(1:end-1,:,k);
            Y_test = temp_y(1:end-1,:,k);
        end
        temp_x(:,:,k) = [];
        temp_y(:,:,k) = [];
        temp_train_x = [];
        temp_train_y = [];
        for s = 1:size(temp_x,3)
            if (sum(temp_x(end,:,s)) ~= 1) && (temp_y(end,:,s) ~= 0)
                temp_train_x = [temp_train_x; temp_x(:,:,s)];
                temp_train_y = [temp_train_y; temp_y(:,:,s)];
            elseif (sum(temp_x(end,:,s)) == 1) || (temp_y(end,:,s) == 0)
                temp_train_x = [temp_train_x; temp_x(1:end-1,:,s)];
                temp_train_y = [temp_train_y; temp_y(1:end-1,:,s)]; 
            end
        end
        [w1, w2, b1, b2] = NN_ReLU_model(temp_train_x, temp_train_y, ...
            init_w1{j}, init_w2{j}, init_b1{j}, init_b2{j}, 20000, 3.5*10^-6);
        predicted_y_test = NN_ReLU_classifier(X_test, w1, w2, b1, b2);
        error_test = mean_squared_error(predicted_y_test, Y_test);
        average_error = [average_error, error_test];
    end
    cross_validation_error = [cross_validation_error, mean(average_error)];
    
    [w1_2,w2_2, b1_2, b2_2] = NN_ReLU_model(X_train_full,y_train_full, ...
        init_w1{j}, init_w2{j}, init_b1{j}, init_b2{j}, 20000, 3.5*10^-6);
    y_pred_train = NN_ReLU_classifier(X_train_full, w1_2, w2_2, b1_2, b2_2);
    error_train_full = mean_squared_error(y_pred_train, y_train_full);
    training_error = [training_error, error_train_full];

    y_pred_test = NN_ReLU_classifier(X_test_full, w1_2, w2_2, b1_2, b2_2);
    error_test_full = mean_squared_error(y_pred_test, y_test_full);
    test_error = [test_error, error_test_full];
end

figure
hold on
plot(initparams,training_error,'r',initparams,test_error,'b',...
    initparams,cross_validation_error)
xlabel('The number of hidden units');
ylabel('Error');
legend('Train error','Test error','Cross-validation error');
hold off

toc