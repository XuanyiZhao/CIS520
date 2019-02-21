function deriv_ReLU_f = derivative_ReLU(f)
% The 1st-order derivative of logistic sigmoid activation function
deriv_ReLU_f = f;
deriv_ReLU_f(deriv_ReLU_f > 0) = 1;
deriv_ReLU_f(deriv_ReLU_f <= 0) = 0;
end