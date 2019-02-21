function [w, b] = LogisticRegressionL2(traindata, trainlabels, lambda)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % lambda      - regularization parameter (positive real number)
        
    % OUTPUT
    % returns learnt model: w - n x 1 weight vector, b - bias term
    initial_w = zeros(58,1);
    L2 = @(w) (1/length(trainlabels)) * sum(log(1+exp(-trainlabels .* traindata * w)))...
        + lambda * (w' * w - w(end) * w(end));
    model = fminunc(L2,initial_w);
    w = model(1:57);
    b = model(58);
end
