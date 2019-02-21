% Read data from 10% to 100% and allocate them to training data and test data
num_train = 250;
num_test = 4351;
X_train_10 = load('../Spambase/Train-subsets/X_train_10%.txt');
X_train_10(:,58) = ones(num_train * 0.1,1);
X_train_20 = load('../Spambase/Train-subsets/X_train_20%.txt');
X_train_20(:,58) = ones(num_train * 0.2,1);
X_train_30 = load('../Spambase/Train-subsets/X_train_30%.txt');
X_train_30(:,58) = ones(num_train * 0.3,1);
X_train_40 = load('../Spambase/Train-subsets/X_train_40%.txt');
X_train_40(:,58) = ones(num_train * 0.4,1);
X_train_50 = load('../Spambase/Train-subsets/X_train_50%.txt');
X_train_50(:,58) = ones(num_train * 0.5,1);
X_train_60 = load('../Spambase/Train-subsets/X_train_60%.txt');
X_train_60(:,58) = ones(num_train * 0.6,1);
X_train_70 = load('../Spambase/Train-subsets/X_train_70%.txt');
X_train_70(:,58) = ones(num_train * 0.7,1);
X_train_80 = load('../Spambase/Train-subsets/X_train_80%.txt');
X_train_80(:,58) = ones(num_train * 0.8,1);
X_train_90 = load('../Spambase/Train-subsets/X_train_90%.txt');
X_train_90(:,58) = ones(num_train * 0.9,1);
X_train_100 = load('../Spambase/Train-subsets/X_train_100%.txt');
X_train_100(:,58) = ones(num_train,1);
Y_train_10 = load('../Spambase/Train-subsets/y_train_10%.txt');
Y_train_20 = load('../Spambase/Train-subsets/y_train_20%.txt');
Y_train_30 = load('../Spambase/Train-subsets/y_train_30%.txt');
Y_train_40 = load('../Spambase/Train-subsets/y_train_40%.txt');
Y_train_50 = load('../Spambase/Train-subsets/y_train_50%.txt');
Y_train_60 = load('../Spambase/Train-subsets/y_train_60%.txt');
Y_train_70 = load('../Spambase/Train-subsets/y_train_70%.txt');
Y_train_80 = load('../Spambase/Train-subsets/y_train_80%.txt');
Y_train_90 = load('../Spambase/Train-subsets/y_train_90%.txt');
Y_train_100 = load('../Spambase/Train-subsets/y_train_100%.txt');
X_test = load('../Spambase/X_test.txt');
X_test(:,58) = ones(num_test,1);
Y_test = load('../Spambase/y_test.txt');
%Training on the 10% data set
[w1,b1]=LogisticRegression(X_train_10,Y_train_10);
training_10_predicted_y = classifier(X_train_10,[w1;b1]);
test_10_predicted_y = classifier(X_test,[w1;b1]);
training_error_10 = classification_error(training_10_predicted_y, Y_train_10);
test_error_10 = classification_error(test_10_predicted_y, Y_test);
%Training on the 20% data set
[w2,b2]=LogisticRegression(X_train_20,Y_train_20);
training_20_predicted_y = classifier(X_train_20,[w2;b2]);
test_20_predicted_y = classifier(X_test,[w2;b2]);
training_error_20 = classification_error(training_20_predicted_y, Y_train_20);
test_error_20 = classification_error(test_20_predicted_y, Y_test);
%Training on the 30% data set
[w3,b3]=LogisticRegression(X_train_30,Y_train_30);
training_30_predicted_y = classifier(X_train_30,[w3;b3]);
test_30_predicted_y = classifier(X_test,[w3;b3]);
training_error_30 = classification_error(training_30_predicted_y, Y_train_30);
test_error_30 = classification_error(test_30_predicted_y, Y_test);
%Training on the 40% data set
[w4,b4]=LogisticRegression(X_train_40,Y_train_40);
training_40_predicted_y = classifier(X_train_40,[w4;b4]);
test_40_predicted_y = classifier(X_test,[w4;b4]);
training_error_40 = classification_error(training_40_predicted_y, Y_train_40);
test_error_40 = classification_error(test_40_predicted_y, Y_test);
%Training on the 50% data set
[w5,b5]=LogisticRegression(X_train_50,Y_train_50);
training_50_predicted_y = classifier(X_train_50,[w5;b5]);
test_50_predicted_y = classifier(X_test,[w5;b5]);
training_error_50 = classification_error(training_50_predicted_y, Y_train_50);
test_error_50 = classification_error(test_50_predicted_y, Y_test);
%Training on the 60% data set
[w6,b6]=LogisticRegression(X_train_60,Y_train_60);
training_60_predicted_y = classifier(X_train_60,[w6;b6]);
test_60_predicted_y = classifier(X_test,[w6;b6]);
training_error_60 = classification_error(training_60_predicted_y, Y_train_60);
test_error_60 = classification_error(test_60_predicted_y, Y_test);
%Training on the 70% data set
[w7,b7]=LogisticRegression(X_train_70,Y_train_70);
training_70_predicted_y = classifier(X_train_70,[w7;b7]);
test_70_predicted_y = classifier(X_test,[w7;b7]);
training_error_70 = classification_error(training_70_predicted_y, Y_train_70);
test_error_70 = classification_error(test_70_predicted_y, Y_test);
%Training on the 80% data set
[w8,b8]=LogisticRegression(X_train_80,Y_train_80);
training_80_predicted_y = classifier(X_train_80,[w8;b8]);
test_80_predicted_y = classifier(X_test,[w8;b8]);
training_error_80 = classification_error(training_80_predicted_y, Y_train_80);
test_error_80 = classification_error(test_80_predicted_y, Y_test);
%Training on the 90% data set
[w9,b9]=LogisticRegression(X_train_90,Y_train_90);
training_90_predicted_y = classifier(X_train_90,[w9;b9]);
test_90_predicted_y = classifier(X_test,[w9;b9]);
training_error_90 = classification_error(training_90_predicted_y, Y_train_90);
test_error_90 = classification_error(test_90_predicted_y, Y_test);
%Training on the 100% data set
[w10,b10]=LogisticRegression(X_train_100,Y_train_100);
training_100_predicted_y = classifier(X_train_100,[w10;b10]);
test_100_predicted_y = classifier(X_test,[w10;b10]);
training_error_100 = classification_error(training_100_predicted_y, Y_train_100);
test_error_100 = classification_error(test_100_predicted_y, Y_test);

%Plotting the learning curve
Training_error = [training_error_10,training_error_20, training_error_30, training_error_40, ...
    training_error_50, training_error_60, training_error_70, training_error_80, ...
    training_error_90, training_error_100];
Test_error = [test_error_10,test_error_20,test_error_30,test_error_40,test_error_50,...
    test_error_60,test_error_70,test_error_80,test_error_90,test_error_100];
Training_subsets_numbers = [25:25:250];
figure
hold on
plot(Training_subsets_numbers,Test_error,'b',Training_subsets_numbers,Training_error,'r')
xlabel('Training subsets numbers');
ylabel('Error');
legend('Test error','Training error');
hold off

% Transfer the input labels from {+1, -1} to {0, 1} so as to do sanity check
Y_train_glm = Y_train_100;
Y_train_glm(Y_train_glm == -1) = 0;
% Sanity checking with glmfit function in MATLAB
glm_model = glmfit(X_train_100, Y_train_glm, 'binomial', 'link', 'logit', 'constant', 'off'); 
glm_training_predicted_y = classifier(X_train_100, glm_model);
glm_test_predicted_y = classifier(X_test, glm_model);
training_error_glm = classification_error(glm_training_predicted_y,Y_train_100); 
test_error_glm = classification_error(glm_test_predicted_y, Y_test);
                                 