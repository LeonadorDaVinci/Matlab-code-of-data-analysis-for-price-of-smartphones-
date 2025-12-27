clc; clear; close all;

%% ------------------ 1. 读取数据 ------------------
filename = 'data3.xlsx';
T1       = readtable(filename, 'Sheet', '性能',     'VariableNamingRule', 'preserve');
T2       = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3       = readtable(filename, 'Sheet', '价格',     'VariableNamingRule', 'preserve');
T_brand  = readtable(filename, 'Sheet', '品牌',     'VariableNamingRule', 'preserve');

%% ------------------ 2. 选取并合并变量 ------------------
% 把 “时间” 和 “首发价格（市场价）” 一起取出
% 假设 T3.时间 已经是 numeric，否则可用 datenum、posixtime 等转为数值
T3_vars = T3(:, {'时间量化','价格差值'});
X_time  = T3_vars.('时间量化');           % N×1 时间自变量
y       = T3_vars.('价格差值');  % N×1 因变量

% 原先的性能、周边配置
T1 = T1(:, {'SoC制程量化','跑分量化'});
T2 = T2(:, {'存储量化','RAM量化','屏幕尺寸量化','分辨率量化', ...
            '刷新率量化','主摄量化','电池容量量化','充电功率量化'});

allData = [T1, T2];
X0      = allData{:, :};         % N×10 其它数值自变量

% 合并时间变量
X0 = [X0, X_time];               % 现在是 N×11

%% ------------------ 3. 生成品牌哑变量 ------------------
brand = categorical(T_brand.Brand);
D_all = dummyvar(brand);         % N×k 全哑变量
D     = D_all(:,1:end-1);        % 丢掉最后一列

X = [X0, D];                     % N×(11 + k−1)
n = size(X, 2);

%% ------------------ 4. 构造 table 及名称映射 ------------------
% 数值自变量名称
numNames   = [T1.Properties.VariableNames, ...
              T2.Properties.VariableNames, {'时间量化'}];  % 10+1 列

% 品牌哑变量名称
brandCats  = categories(brand);
dummyRaw   = strcat('Brand_', brandCats(1:end-1));      % k−1 列

respName   = 'Price';

origNames  = [numNames, dummyRaw', {respName}];         % 总列数

validNames = matlab.lang.makeValidName(origNames);
validNames = matlab.lang.makeUniqueStrings(validNames);

tbl = array2table([X, y], 'VariableNames', validNames);

%% ------------------ 5. 拟合高斯模型 ------------------
predictors = validNames(1:end-1);
response   = validNames{end};

% 定义高斯模型
gaussianModel = @(p, X) p(1) * exp(-sum(p(2:n+1) .* (X - p(n+2:end)).^2, 2));

% 初始值
a0      = mean(y);
b0      = ones(1, n) * 0.01;
c0      = mean(X);
beta0   = [a0, b0, c0];

% 非线性拟合
mdl = fitnlm(X, y, gaussianModel, beta0);
disp(mdl)

%% ------------------ 6. 显示结果和误差分析 ------------------
% 打印映射
fprintf('变量名映射 (Clean -> Original):\n');
for i = 1:numel(validNames)
    fprintf('  %s -> %s\n', validNames{i}, origNames{i});
end
fprintf('\n');

% 提取系数
b = mdl.Coefficients.Estimate;

% 显示拟合函数
fprintf('拟合函数：\n%s = %.4f × exp( - [', respName, b(1));
for i = 1:n
    fprintf('%.4f·(%s - %.4f)^2', b(1+i), origNames{i}, b(1+n+i));
    if i < n, fprintf(' + '); end
end
fprintf('] )\n\n');

% 构造预测函数
f = @(x) b(1) * exp(-sum(b(2:n+1) .* (x - b(n+2:end)).^2, 2));

% 示例预测
x1    = X(1, :);
pred1 = f(x1);
fprintf('第1条样本预测价格：%.2f\n', pred1);

% 全部预测 & 误差
y_pred   = predict(mdl, X);
residual = y_pred - y;

meanErr = mean(residual);
maxErr  = max(residual);
minErr  = min(residual);

fprintf('平均误差 = %.2f（负值表示整体偏低）\n', meanErr);
fprintf('最大误差 = %.2f，最小误差 = %.2f\n', maxErr, minErr);

% 可视化
figure;
subplot(1,2,1);
plot(y, y_pred, 'bo'); hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('实际值'); ylabel('预测值'); title('高斯拟合：预测 vs 实际');
grid on; axis equal;

subplot(1,2,2);
histogram(residual, 20);
xlabel('误差'); ylabel('频数'); title('残差分布');
grid on;
