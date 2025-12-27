clc; clear; close all;

%% ------------------ 1. 读取数据 ------------------
filename = 'data3.xlsx';
T1       = readtable(filename, 'Sheet', '性能',     'VariableNamingRule', 'preserve');
T2       = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3       = readtable(filename, 'Sheet', '价格',     'VariableNamingRule', 'preserve');
T_brand  = readtable(filename, 'Sheet', '品牌',     'VariableNamingRule', 'preserve');

%% ------------------ 2. 选取并合并变量 ------------------
T1 = T1(:, {'SoC制程量化','跑分量化'});
T2 = T2(:, {'存储量化','RAM量化','屏幕尺寸量化','分辨率量化', ...
            '刷新率量化','主摄量化','电池容量量化','充电功率量化'});
T3 = T3(:, {'价格差值'});    % 因变量

allData = [T1, T2, T3];
X0      = allData{:,1:end-1};     % N×10 数值自变量
y       = allData{:,end};         % N×1 因变量

%% ------------------ 3. 生成品牌哑变量 ------------------
brand = categorical(T_brand.Brand);
D_all = dummyvar(brand);          % N×k 全哑变量
D     = D_all(:,1:end-1);         % 丢掉最后一列，保留 k−1 列

X = [X0, D];                      % N×(10+k−1)
n = size(X, 2);

%% ------------------ 4. 构造 table 及名称映射 ------------------
numNames   = [T1.Properties.VariableNames, T2.Properties.VariableNames];  % 10 列
brandCats  = categories(brand);
dummyRaw   = strcat('Brand_', brandCats(1:end-1));                       % k−1 列
respName   = 'Price';                                                     % 因变量名

origNames  = [numNames, dummyRaw', {respName}];  % 总共 10+(k−1)+1 列

% 生成合法且唯一的 MATLAB 名称
validNames = matlab.lang.makeValidName(origNames);
validNames = matlab.lang.makeUniqueStrings(validNames);

tbl = array2table([X, y], 'VariableNames', validNames);

%% ------------------ 5. 拟合高斯模型 ------------------
predictors = validNames(1:end-1);
response   = validNames{end};

% 高斯模型定义
% y = a * exp( -sum( bi * (xi - ci)^2 ) )
% p = [a, b1...bn, c1...cn]
gaussianModel = @(p, X) p(1) * exp(-sum(p(2:n+1) .* (X - p(n+2:end)).^2, 2));

% 初始值设定
a0 = mean(y);
b0 = ones(1, n) * 0.01;
c0 = mean(X);
beta0 = [a0, b0, c0];

% 非线性拟合
mdl = fitnlm(X, y, gaussianModel, beta0);

disp(mdl)

%% ------------------ 6. 显示结果和误差分析 ------------------
% 打印变量映射
fprintf('变量名映射 (Clean -> Original):\n');
for i = 1:numel(validNames)
    fprintf('  %s -> %s\n', validNames{i}, origNames{i});
end
fprintf('\n');

% 系数向量
b = mdl.Coefficients.Estimate;

% 拟合函数显示（使用原始变量名）
fprintf('拟合函数格式：\n');
fprintf('%s = %.4f × exp( - [', respName, b(1));
for i = 1:n
    fprintf('%.4f·(%s - %.4f)^2', b(1+i), origNames{i}, b(1+n+i));
    if i < n
        fprintf(' + ');
    end
end
fprintf('] )\n\n');

% 构造预测函数
f = @(x) b(1) * exp(-sum(b(2:n+1) .* (x - b(n+2:end)).^2, 2));

% 示例预测
x1    = X(1, :);
pred1 = f(x1);
fprintf('第1条样本预测价格：%.2f\n', pred1);

% 全部预测
y_pred   = predict(mdl, X);
residual = y_pred - y;

% 误差分析
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
