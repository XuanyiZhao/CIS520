function [a, b] = RBF_kernel_svm_model(traindata, trainlabels, C, gamma)
% X = quadprog(H,f,A,b,Aeq,beq,LB,UB,X0,OPTIONS)
m = length(trainlabels);
H = zeros(m, m);
for i = 1:m
    H(:,i) = exp((-sum((traindata - traindata(i,:)) .^ 2, 2)) * gamma);
end
H = (trainlabels * trainlabels') .* H;

f = ones(m, 1);
Aeq = trainlabels';
beq = 0;
LB = zeros(m, 1);
UB = ones(m, 1) * C;
options = optimoptions('quadprog', 'MaxIterations', 100000);
a = quadprog(H, -f, [], [], Aeq, beq, LB, UB, [], options);
a(a < 0.000001 * C) = 0;
a(a > (C - C * 0.000001)) = C;
sv = find(a > 0.000001 * C);
sv1 = find(0.000001 * C < a < (C - C * 0.000001));

b = 0;
for k = 1:length(sv1)
    temp = 0;
    for j = 1:length(sv)
        temp = temp + a(sv(j)) * trainlabels(sv(j)) * ...
        exp((-sum((traindata(sv1(k),:) - traindata(sv(j),:)) .^ 2, 2)) * gamma);
    end
b = b + trainlabels(sv1(k)) - temp;
end
b = b / length(sv1);

end