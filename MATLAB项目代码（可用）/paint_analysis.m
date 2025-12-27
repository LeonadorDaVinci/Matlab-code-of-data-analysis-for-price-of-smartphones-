% 假设：
% X           - nSamples × nFeatures，标准化或原始特征矩阵
% coeff       - PCA载荷矩阵 (nFeatures × pcNum)
% b_opt       - 高斯部分 b 参数 (pcNum × 1)
% c_opt       - 高斯部分 c 参数 (pcNum × 1)
% a_opt       - 高斯部分 a 参数 (标量)
% p_opt_linear - 线性部分权重 (含截距，(nFeatures+1) × 1)
% varNames    - 长度为 nFeatures 的变量名 cell 数组，例如： {'SoC', '跑分', '存储', ...}

% 1. 线性部分预测
intercept = p_opt_linear(1);
beta = p_opt_linear(2:end);
L_pred = X * beta + intercept;

% 2. 高斯部分预测
X_mean = mean(X);
X_std = std(X);
Xn = (X - X_mean) ./ X_std;  % 标准化，跟训练时一致

pcNum = length(b_opt);
Z = Xn * coeff(:,1:pcNum);

diffs = (Z - c_opt').^2;
exp_terms = exp(- diffs * b_opt);
G_pred = a_opt * exp_terms;

% 3. 总预测
y_pred = L_pred + G_pred;

% 4. 贡献比例（用协方差）
var_total = var(y_pred);
cov_L = cov(y_pred, L_pred);
cov_G = cov(y_pred, G_pred);

contrib_L = cov_L(1,2) / var_total;
contrib_G = cov_G(1,2) / var_total;

fprintf('线性部分贡献比例：%.2f%%\n', contrib_L*100);
fprintf('高斯部分贡献比例：%.2f%%\n', contrib_G*100);

% 5. 贡献率柱状图
figure;
bar([contrib_L, contrib_G]*100);
set(gca, 'xticklabel', {'线性部分', '高斯部分'});
ylabel('贡献率 (%)');
title('线性与高斯部分对预测的贡献');
grid on;

% 6. 线性权重排序与变量名（绝对值排序，前10）
[sorted_beta, idx_beta] = sort(abs(beta), 'descend');
topN = min(10, length(beta));
figure;
bar(sorted_beta(1:topN));
set(gca, 'xticklabel', varNames(idx_beta(1:topN)));
xtickangle(45);
ylabel('权重绝对值');
title('线性部分变量权重排序（前10）');
grid on;

% 7. 高斯部分主成分 b 参数排序（绝对值排序，前10）
[sorted_b, idx_b] = sort(abs(b_opt), 'descend');
figure;
bar(sorted_b(1:min(10,length(b_opt))));
set(gca, 'xticklabel', cellstr(string(idx_b(1:min(10,length(b_opt))))));
xtickangle(45);
ylabel('b 参数绝对值');
title('高斯部分主成分权重排序（前10）');
grid on;

