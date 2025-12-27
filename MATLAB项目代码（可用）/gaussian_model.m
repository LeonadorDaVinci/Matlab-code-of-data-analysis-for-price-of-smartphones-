function ypred = gaussian_model(p, Xdata, nVars_local)
% GAUSSIAN_MODEL  高斯模型预测
%   ypred = gaussian_model(p, Xdata, nVars_local)
%   p: [a; b(1..n); c(1..n)]
a = p(1);
b = p(2:1+nVars_local);
c = p(2+nVars_local:end);
% 用显式 repmat 保证兼容旧版本 MATLAB
diffs = (Xdata - repmat(c', size(Xdata,1), 1)).^2;
expo  = exp( - diffs * b );
ypred = a * expo;
end


