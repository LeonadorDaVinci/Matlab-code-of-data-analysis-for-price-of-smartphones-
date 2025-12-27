clc; clear;

% 读取数据
filename = 'data3.xlsx';
T1 = readtable(filename, 'Sheet', '性能', 'VariableNamingRule', 'preserve');
T2 = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3 = readtable(filename, 'Sheet', '价格', 'VariableNamingRule', 'preserve');

% 选取需要的列
T1 = T1(:, {'SoC制程量化','跑分量化'});  % 2 列
T2 = T2(:, {'存储量化','RAM量化','屏幕尺寸量化','分辨率量化', ...
            '刷新率量化','主摄量化','电池容量量化','充电功率量化'}); % 8 列
T3 = T3(:, {'首发价格（市场价）'});       % 因变量

fprintf('T1 行数: %d\n', height(T1));
fprintf('T2 行数: %d\n', height(T2));
fprintf('T3 行数: %d\n\n', height(T3));
% 合并数据
allData = [T1, T2, T3];

% 构造自变量矩阵 X (N×10) 和 因变量 y (N×1)
X = allData{:,1:end-1};
y = allData{:,end};

% 定义多元高斯模型（假设协方差为对角阵）
% 参数向量 b 的含义：
%   b(1)         = 振幅 A
%   b(2:11)      = mu 向量（10×1）
%   b(12:21)     = sigma 向量（10×1）
modelfun = @(b, X) b(1) .* exp(-0.5 * sum(((X - b(2:11)) ./ b(12:21)).^2, 2));

% 初始猜测：幅值取 y 最大值，mu 取 X 各列均值，sigma 取 X 各列标准差
beta0 = [ max(y), mean(X), std(X) ];

% 非线性最小二乘拟合
beta = nlinfit(X, y, modelfun, beta0);

% 拆分拟合参数
A     = beta(1);
mu    = beta(2:11);
sigma = beta(12:21);

% 构造最终的拟合函数
fitFunc = @(x) A .* exp(-0.5 * sum(((x - mu) ./ sigma).^2, 2));

% 显示结果
fprintf('拟合振幅 A = %.6f\n', A);
disp('拟合均值向量 μ =');  disp(mu);
disp('拟合标准差向量 σ ='); disp(sigma);

fprintf('\n最终拟合函数 f(x) 为：\n');
fprintf('f(x) = %.6f * exp(-0.5 * Σ_{i=1}^{10} ((x_i - %.6f)/%.6f)^2 )\n', ...
        A, mu(1), sigma(1));
for i = 2:10
    fprintf('                    ((x_%d - %.6f)/%.6f)^2 +\n', i, mu(i), sigma(i));
end
fprintf('                    )\n\n');

% 误差分析

y_pred = fitFunc(X);

% 残差（预测值 - 实际值）
residuals = y_pred - y;

% 均方根误差 RMSE
rmse = sqrt(mean(residuals.^2));

% 平均绝对误差 MAE
mae = mean(abs(residuals));

% 决定系数 R²
SS_res = sum((y - y_pred).^2);
SS_tot = sum((y - mean(y)).^2);
R2 = 1 - SS_res / SS_tot;

% 最大误差和最小误差
max_error = max(residuals);
min_error = min(residuals);

% 输出误差分析结果
fprintf('--- 误差分析 ---\n');
fprintf('均方根误差 RMSE = %.4f\n', rmse);
fprintf('平均绝对误差 MAE = %.4f\n', mae);
fprintf('R² 决定系数 = %.4f\n', R2);
fprintf('最大误差 = %.2f\n', max_error);
fprintf('最小误差 = %.2f\n', min_error);

figure;
subplot(1,2,1);
plot(y, y_pred, 'bo'); hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('实际值'); ylabel('预测值'); title('预测 vs 实际');
grid on; axis equal;

subplot(1,2,2);
histogram(residuals, 20);
xlabel('误差'); ylabel('频数'); title('残差分布');
grid on;