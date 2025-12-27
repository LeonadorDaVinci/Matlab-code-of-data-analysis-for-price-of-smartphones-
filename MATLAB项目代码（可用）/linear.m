clc; clear; close all;

%% ------------------ 1. 读取数据 ------------------
filename = 'data3.xlsx';
T1       = readtable(filename, 'Sheet', '性能',     'VariableNamingRule', 'preserve');
T2       = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3       = readtable(filename, 'Sheet', '价格',     'VariableNamingRule', 'preserve');
T_brand  = readtable(filename, 'Sheet', '品牌',     'VariableNamingRule', 'preserve');

%% ------------------ 2. 选取并合并变量 ------------------
% 把 “时间” 和 “价格差值” 一起取出（假设时间已被量化为数值）
T3_vars = T3(:, {'时间量化','价格差值'});
X_time  = T3_vars.('时间量化');           % N×1 时间自变量
y       = T3_vars.('价格差值');           % N×1 因变量

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
D     = D_all(:,1:end-1);        % 丢掉最后一列（避免虚拟变量陷阱）

X = [X0, D];                     % N×(11 + k−1)
n = size(X, 2);

%% ------------------ 4. 构造 table 及名称映射 ------------------
% 数值自变量名称（11 列）
numNames   = [T1.Properties.VariableNames, ...
              T2.Properties.VariableNames, {'时间量化'}];  % 10+1 列

% 品牌哑变量名称
brandCats  = categories(brand);
dummyRaw   = strcat('Brand_', brandCats(1:end-1));      % k−1 列

respName   = 'Price';  % 响应变量的清洁名（用于表头）

origNames  = [numNames, dummyRaw', {respName}];         % 总列名（原始映射）

validNames = matlab.lang.makeValidName(origNames);
validNames = matlab.lang.makeUniqueStrings(validNames);

tbl = array2table([X, y], 'VariableNames', validNames);

%% ------------------ 5. 线性拟合（普通最小二乘，含截距） ------------------
predictors = validNames(1:end-1);
response   = validNames{end};

% 使用 table 中的所有自变量拟合线性回归（含截距）
mdl = fitlm(tbl, 'ResponseVar', response);

% 显示模型摘要（含 R^2、F 统计量、系数表等）
disp(mdl)

%% ------------------ 6. 显示结果和误差分析 ------------------
% 打印映射（Clean -> Original）
fprintf('变量名映射 (Clean -> Original):\n');
for i = 1:numel(validNames)
    fprintf('  %s -> %s\n', validNames{i}, origNames{i});
end
fprintf('\n');

% 提取系数（第一项为截距）
b = mdl.Coefficients.Estimate;
coefNames = mdl.CoefficientNames;  % 与 b 对应的系数名

% 显示线性回归系数
fprintf('线性回归系数（含截距）：\n');
for i = 1:numel(b)
    fprintf('  %s : %.6f\n', coefNames{i}, b(i));
end
fprintf('\n');

% 打印为一行方程（便于阅读）
fprintf('%s = %.6f', respName, b(1)); % 截距
for i = 2:numel(b)
    fprintf(' + (%.6f)*%s', b(i), coefNames{i});
end
fprintf('\n\n');

% ---------- 示例预测（使用 predict，推荐） ----------
pred1 = predict(mdl, tbl(1,:));   % 使用 fitlm 的 predict（自动处理 table）
fprintf('第1条样本预测价格（predict）: %.2f\n', pred1);

% ---------- 可选：使用系数 b 的自定义预测函数 ----------
% 如果你想直接用系数 b 做预测，可以调用下面的局部函数 predict_from_b
% 它兼容行向量、列向量或 m×n 矩阵。
pred1_alt = predict_from_b(X(1,:), b);   % 传入行向量（1×n）
fprintf('第1条样本预测价格（predict_from_b）: %.2f\n', pred1_alt);

% 全部预测 & 误差
y_pred   = predict(mdl, tbl);   % N×1
residual = y_pred - y;

meanErr = mean(residual);
maxErr  = max(residual);
minErr  = min(residual);

fprintf('平均误差 = %.2f（正值表示整体偏高）\n', meanErr);
fprintf('最大误差 = %.2f，最小误差 = %.2f\n', maxErr, minErr);

% 可视化：预测 vs 实际 与 残差分布
figure;
subplot(1,2,1);
plot(y, y_pred, 'bo'); hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('实际值'); ylabel('预测值'); title('线性拟合：预测 vs 实际');
grid on; axis equal;

subplot(1,2,2);
histogram(residual, 20);
xlabel('误差'); ylabel('频数'); title('残差分布');
grid on;

%% ------------------ 局部函数：用系数 b 做预测（兼容行/列/矩阵） ------------------
function yhat = predict_from_b(x, b)
% x: 1×n 行向量，或 n×1 列向量，或 m×n 矩阵（m 个样本）
% b: 系数向量（第一项为截距，后面为各特征系数）
    if isvector(x)
        if isrow(x)
            yhat = b(1) + x * b(2:end);
        else
            % x 为列向量 n×1
            yhat = b(1) + (b(2:end)' * x);
        end
    else
        % x 为 m×n 矩阵（每行为一个样本）
        yhat = b(1) + x * b(2:end);
    end
end
