function [a, b] = kernel_svm_model(traindata, trainlabels, C, q)
% X = quadprog(H,f,A,b,Aeq,beq,LB,UB,X0,OPTIONS)
H = trainlabels .* ((traindata * traindata' + 1) .^ q) .* trainlabels';
m = length(trainlabels);
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

sum_diff = 0;
for i = 1:length(sv1)
    diff = trainlabels(sv1(i)) - a(sv)' * (trainlabels(sv) .* ((traindata(sv,:) * traindata(sv1(i),:)' + 1).^q));
    sum_diff = sum_diff + diff;
end
b = sum_diff / length(sv1);

end