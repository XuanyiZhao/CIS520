X_train_100 = load('../Spambase/Train-subsets/X_train_100%.txt');
X_train_100(:,58) = ones(250,1);
Y_train_100 = load('../Spambase/Train-subsets/y_train_100%.txt');
X_test100 = load('../Spambase/X_test.txt');
X_test100(:,58) = ones(4351,1);
Y_test100 = load('../Spambase/y_test.txt');

train_error = [];
test_error = [];
value_of_lambda = [];
for l = 0:7
    lambda = 10^(l-7);
    value_of_lambda = [value_of_lambda, lambda];
    [w_full,b_full] = LogisticRegressionL2(X_train_100,Y_train_100,lambda);
    y_predicted_train_full = classifier(X_train_100,[w_full;b_full]);
    y_predicted_test_full = classifier(X_test100,[w_full;b_full]);
    train_error = [train_error, classification_error(y_predicted_train_full,Y_train_100)];
    test_error = [test_error, classification_error(y_predicted_test_full,Y_test100)];
end

figure
hold on
plot([-7,-6,-5,-4,-3,-2,-1,0],train_error,'r',[-7,-6,-5,-4,-3,-2,-1,0],test_error,'b')
xlabel('Value of lambda');
ylabel('Error');
legend('Train error','Test error');
hold off

for i = 1:5
    X_train(:,:,i) = [load(['../Spambase/Cross-validation/Fold',num2str(i),'/X.txt']), ones(50,1)];
    Y_train(:,:,i) = load(['../Spambase/Cross-validation/Fold',num2str(i),'/y.txt']);
end

cross_validation_error = [];
for j = 0:7
    average_error = [];
    lambda = 10^(j-7);
    for k = 1:5
        temp_x = X_train;
        temp_y = Y_train;
        X_test = temp_x(:,:,k);
        Y_test = temp_y(:,:,k);
        temp_x(:,:,k) = [];
        temp_y(:,:,k) = [];
        temp_train_x = [];
        temp_train_y = [];
        for s = 1:size(temp_x,3)
            temp_train_x = [temp_train_x; temp_x(:,:,s)];
            temp_train_y = [temp_train_y; temp_y(:,:,s)];
        end
        [w,b] = LogisticRegressionL2(temp_train_x,temp_train_y,lambda);
        y_predicted_test = classifier(X_test,[w;b]);
        error_test = classification_error(y_predicted_test,Y_test);
        average_error = [average_error, error_test];
    end
    cross_validation_error = [cross_validation_error, mean(average_error)];
end

