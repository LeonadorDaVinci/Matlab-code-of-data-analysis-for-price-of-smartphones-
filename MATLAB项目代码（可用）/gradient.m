clc; clear; close all;

%% ------------------ 1. 读取数据 ------------------
filename = 'data3.xlsx';
T1       = readtable(filename, 'Sheet', '性能',     'VariableNamingRule', 'preserve');
T2       = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3       = readtable(filename, 'Sheet', '价格',     'VariableNamingRule', 'preserve');
T_brand  = readtable(filename, 'Sheet', '品牌',     'VariableNamingRule', 'preserve');

%% ------------------ 2. 选取并合并变量 ------------------
T3_vars = T3(:, {'时间量化','价格差值'});
X_time  = T3_vars.('时间量化');           
y       = T3_vars.('价格差值');           

T1 = T1(:, {'SoC制程量化','跑分量化'});
T2 = T2(:, {'存储量化','RAM量化','屏幕尺寸量化','分辨率量化', ...
            '刷新率量化','主摄量化','电池容量量化','充电功率量化'});
allData = [T1, T2];
X0      = allData{:, :};          
X0      = [X0, X_time];          

brand = categorical(T_brand.Brand);
D_all = dummyvar(brand);         
D     = D_all(:,1:end-1);        
X     = [X0, D];                 

[nSamples, nVars] = size(X);

%% ------------------ 3. 梯度下降拟合高斯模型 ------------------
% 模型：f_p(x) = a * exp( - sum_{i=1}^n b_i * (x_i - c_i)^2 )
% 参数向量 p = [a, b1...bn, c1...cn]，长度 1 + nVars + nVars
m = 1 + nVars + nVars;
% 初始化
p = zeros(m,1);
p(1) = mean(y);            % a 初始
p(2:1+nVars) = 0.01;       % b_i 初始
p(2+nVars:end) = mean(X);  % c_i 初始

alpha = 1e-13;             % 学习率，可根据情况调节
numIter = 1900000;           % 迭代次数

% 记录损失
lossHistory = zeros(numIter,1);

for k = 1:numIter
    % 计算预测
    a = p(1);
    b = p(2:1+nVars);
    c = p(2+nVars:end);
    % 逐样本计算
    diffs = (X - c').^2;               % N×nVars
    expo  = exp(- diffs * b );        % N×1
    y_pred = a * expo;                % N×1

    % 计算均方误差和梯度
    err = y_pred - y;                 % N×1
    loss = mean(err.^2);
    lossHistory(k) = loss;

    % 参数梯度
    % ∂J/∂a = (2/N) * sum[ err * exp(...) ]
    grad_a = (2/ nSamples) * sum( err .* expo );
    % ∂J/∂b_i = (2/N) * sum[ err * a * exp(...) * ( - (x_i - c_i)^2 ) ]
    grad_b = zeros(nVars,1);
    for i = 1:nVars
        grad_b(i) = (2/ nSamples) * sum( err .* a .* expo .* ( - diffs(:,i) ) );
    end
    % ∂J/∂c_i = (2/N) * sum[ err * a * exp(...) * ( - b_i * 2 * (x_i - c_i) ) ]
    grad_c = zeros(nVars,1);
    for i = 1:nVars
        grad_c(i) = (2/ nSamples) * sum( err .* a .* expo .* ( -2 * b(i) .* (X(:,i) - c(i)) ) );
    end

    % 更新参数
    p(1)               = p(1)               - alpha * grad_a;
    p(2:1+nVars)       = p(2:1+nVars)       - alpha * grad_b;
    p(2+nVars:end)     = p(2+nVars:end)     - alpha * grad_c;
end

% 最终参数
a_opt = p(1);
b_opt = p(2:1+nVars);
c_opt = p(2+nVars:end);

%% ------------------ 4. 结果展示 ------------------
fprintf('梯度下降拟合结束：\n');
fprintf('  a = %.4f\n', a_opt);
for i = 1:nVars
    fprintf('  b(%d) = %.4f, c(%d) = %.4f\n', i, b_opt(i), i, c_opt(i));
end

% 预测与误差
y_pred_final = a_opt * exp(- sum( b_opt'.*(X - c_opt').^2, 2 ));
residual     = y_pred_final - y;
fprintf('平均误差 = %.2f， 最大误差 = %.2f， 最小误差 = %.2f\n', ...
        mean(residual), max(residual), min(residual));

% 损失曲线
figure;
plot(1:numIter, lossHistory, 'LineWidth',1.5);
xlabel('迭代次数');
ylabel('均方误差 Loss');
title('梯度下降收敛曲线');
grid on;

% 拟合效果散点图
figure;
subplot(1,2,1);
plot(y, y_pred_final, 'bo'); hold on;
plot([min(y),max(y)], [min(y),max(y)], 'r--');
xlabel('实际'); ylabel('预测'); title('高斯模型：预测 vs 实际');
axis equal; grid on;

subplot(1,2,2);
histogram(residual,20);
xlabel('残差'); ylabel('频数'); title('残差分布');
grid on;
