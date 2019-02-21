function [res] = mean_squared_error(y_pred, y_true)
% This function computes the MSE for the predicted labels with respect to 
% the ground truth. The returned error value is a real number.

% y_true: vector of true labels (real numbers)
% y_pred: vector of predicted labels (real numbers)
% err: mean squared error

    res = sum((y_true - y_pred).^2) / length(y_true);
end