function [w, b] = LogisticRegression(traindata, trainlabels)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data    
    
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    
    % Fill in your code here    
    % Consider using fminunc MATLAB function for solving the logistic regression optimization problem.
    initial_w = zeros(58,1);
    negative_log_likelihood = @(w) sum(log(1+exp(-trainlabels .* traindata * w)));
    options = optimoptions(@fminunc,'Algorithm','quasi-newton','MaxFunctionEvaluations', 30000, 'MaxIterations', 1000);
    model = fminunc(negative_log_likelihood,initial_w,options);
    w = model(1:57);
    b = model(58);
end
