num_train = 721;
num_test = 309;
X_train_10 = load('../Data-set-2/Train-subsets/X_train_10%.txt');
X_train_10(:,9) = ones(round(num_train * 0.1),1);
Y_train_10 = load('../Data-set-2/Train-subsets/y_train_10%.txt');
X_test100 = load('../Data-set-2/X_test.txt');
X_test100(:,9) = ones(num_test,1);
Y_test100 = load('../Data-set-2/y_test.txt');

train_error_10 = [];
test_error_10 = [];
value_of_lambda = [];
lambda_vector = [0.1, 1, 10, 100, 500, 1000];
I = eye(8);
I(:,9)=zeros(8,1);
I(9,:)=zeros(9,1);
for l = 1:6
    lambda = lambda_vector(l);
    value_of_lambda = [value_of_lambda, lambda];
    w1_b1 = (X_train_10' * X_train_10 + lambda * round(num_train * 0.1) * I)^-1 ...
        * X_train_10' * Y_train_10;
    predicted_y_train10 = X_train_10 * w1_b1;
    predicted_y_test10 = X_test100 * w1_b1;
    train_error_10 = [train_error_10, mean_squared_error(Y_train_10, predicted_y_train10)];
    test_error_10 = [test_error_10, mean_squared_error(Y_test100, predicted_y_test10)];
end

for i = 1:2
    X_train(:,:,i) = [load(['../Data-set-2/Cross-validation/Fold',num2str(i),'/X_10%.txt']), ones(15,1)];
    Y_train(:,:,i) = load(['../Data-set-2/Cross-validation/Fold',num2str(i),'/y_10%.txt']);
end

for i = 3:5
    X_train(:,:,i) = [[load(['../Data-set-2/Cross-validation/Fold',num2str(i),'/X_10%.txt']); zeros(1,8)], ones(15,1)];
    Y_train(:,:,i) = [load(['../Data-set-2/Cross-validation/Fold',num2str(i),'/y_10%.txt']); zeros(1,1)];
end

cross_validation_error = [];
for j = 1:6
    average_error = [];
    lambda = lambda_vector(j);
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
        w_b = (temp_train_x' * temp_train_x + lambda * round(num_train * 0.1) * I)^-1 ...
        * temp_train_x' * temp_train_y;
        predicted_y_test = X_test * w_b;
        error_test = mean_squared_error(Y_test,predicted_y_test);
        average_error = [average_error, error_test];
    end
    cross_validation_error = [cross_validation_error, mean(average_error)];
end

figure
hold on
plot([-1, 0, 1, 2, log10(500), 3],train_error_10,'r',[-1, 0, 1, 2, log10(500), 3],test_error_10,'b',...
    [-1, 0, 1, 2, log10(500), 3],cross_validation_error)
xlabel('Log10(Value of lambda)');
ylabel('Error');
legend('Train error','Test error','Cross-validation error');
hold off