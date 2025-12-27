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
T3 = T3(:, {'首发价格（市场价）'});    % 因变量

allData = [T1, T2, T3];
X0      = allData{:,1:end-1};     % N×10 数值自变量
y       = allData{:,end};         % N×1 因变量

%% ------------------ 3. 生成品牌哑变量 ------------------
brand = categorical(T_brand.Brand);
D_all = dummyvar(brand);          % N×k 全哑变量
D     = D_all(:,1:end-1);         % 丢掉最后一列，保留 k−1 列

X = [X0, D];                      % N×(10+k−1)

%% ------------------ 4. 构造 table 及名称映射 ------------------
% 原始变量名列表
numNames   = [T1.Properties.VariableNames, T2.Properties.VariableNames];  % 10 列
brandCats  = categories(brand);
dummyRaw   = strcat('Brand_', brandCats(1:end-1));                       % k−1 列
respName   = 'Price';                                                     % 因变量名

origNames  = [numNames, dummyRaw', {respName}];  % 总共 10+(k−1)+1 列

% 生成合法且唯一的 MATLAB 名称
validNames = matlab.lang.makeValidName(origNames);
validNames = matlab.lang.makeUniqueStrings(validNames);

% 构造表格
tbl = array2table([X, y], 'VariableNames', validNames);

%% ------------------ 5. 拟合线性模型 ------------------
predictors = validNames(1:end-1);
response   = validNames{end};

mdl = fitlm(tbl, 'ResponseVar', response, 'PredictorVars', predictors);

disp(mdl.Coefficients)
disp(anova(mdl, 'summary'))

%% ------------------ 6. 输出映射关系和拟合函数 ------------------
% 打印映射
fprintf('变量名映射 (Clean -> Original):\n');
for i = 1:numel(validNames)
    fprintf('  %s -> %s\n', validNames{i}, origNames{i});
end
fprintf('\n');

% 提取系数
b         = mdl.Coefficients.Estimate;
coefNames = mdl.CoefficientNames;

% 打印可读的拟合函数（使用原始名称）
fprintf('拟合函数：\n%s = %.4f', respName, b(1));
for i = 2:numel(b)
    idx = find(strcmp(validNames, coefNames{i}));
    fprintf(' %+.4f×%s', b(i), origNames{idx});
end
fprintf('\n\n');

% 构造匿名预测函数
f = @(x) b(1) + x * b(2:end);

% 示例预测
x1    = tbl{1, predictors};
pred1 = f(x1);
fprintf('第1条样本预测价格：%.2f\n', pred1);

y_pred   = predict(mdl, tbl);      
y_actual = tbl.Price;               

% 计算残差：预测值 – 真实值
residual = y_pred - y_actual;       

% 查看平均误差、最大/最小误差
meanErr = mean(residual);
maxErr  = max(residual);
minErr  = min(residual);

fprintf('平均误差 = %.2f（负值表示整体偏低）\n', meanErr);
fprintf('最大误差 = %.2f，最小误差 = %.2f\n', maxErr, minErr);

figure;
subplot(1,2,1);
plot(y, y_pred, 'bo'); hold on;
plot([min(y), max(y)], [min(y), max(y)], 'r--');
xlabel('实际值'); ylabel('预测值'); title('预测 vs 实际');
grid on; axis equal;

subplot(1,2,2);
histogram(residual, 20);
xlabel('误差'); ylabel('频数'); title('残差分布');
grid on;