
% PCA + L2 正则化的高斯拟合（lsqnonlin）
% 运行：在命令行输入 pca_reg_gauss_fit

clc; clear; close all;

%% 1. 读取 & 合并变量（和你原来一致）
filename = 'data3.xlsx';
T1       = readtable(filename, 'Sheet', '性能',     'VariableNamingRule', 'preserve');
T2       = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3       = readtable(filename, 'Sheet', '价格',     'VariableNamingRule', 'preserve');
T_brand  = readtable(filename, 'Sheet', '品牌',     'VariableNamingRule', 'preserve');

T3_vars = T3(:, {'时间量化','价格差值'});
X_time  = T3_vars.('时间量化');
y       = T3_vars.('价格差值');

T1 = T1(:, {'SoC制程量化','跑分量化'});
T2 = T2(:, {'存储量化','RAM量化','屏幕尺寸量化','分辨率量化', ...
            '刷新率量化','主摄量化','电池容量量化','充电功率量化'});
allData = [T1, T2];
X0      = allData{:, :};
X0      = [X0, X_time];

% 品牌虚拟变量（合并低频品牌为 "Other" 的建议在下方有代码）
brand = categorical(T_brand.Brand);
D_all = dummyvar(brand);
D = D_all(:,1:(size(D_all,2)-1));   % 去掉一列
X = [X0, D];

[nSamples, nVars] = size(X);
fprintf('原始自变量维度: %d 样本 x %d 变量\n', nSamples, nVars);

%% 2. 基础检测：列方差 / 常数列 / condition number
colStd = std(X,0,1);
zeroCols = find(colStd < 1e-8);
if ~isempty(zeroCols)
    fprintf('警告：发现 %d 列近似常数，将在 PCA 前移除这些列（索引示例）。\n', numel(zeroCols));
    disp(zeroCols);
    X(:,zeroCols) = [];
    nVars = size(X,2);
end
% 简单 condition number 检查（对标准化后的矩阵）
Xn_check = (X - mean(X,1)) ./ max(1e-8, std(X,0,1));
condNum = cond(Xn_check'*Xn_check);
fprintf('X^T X 条件数 (近似)： %.3e\n', condNum);

%% 3. 标准化并 PCA（保留 >=95% 累计方差，若太少则保留最少 10 PC）
muX = mean(X,1);
sigmaX = std(X,0,1); sigmaX(sigmaX==0)=1;
Xn = (X - muX) ./ sigmaX;

[coeff, score, latent, ~, explained] = pca(Xn);
cumexp = cumsum(explained);
pcNum = find(cumexp >= 95, 1, 'first');
if isempty(pcNum), pcNum = min(10, size(Xn,2)); end
pcNum = max(pcNum, min(10, size(Xn,2))); % 最少保留10个或数据维度
fprintf('PCA 后保留主成分数 pcNum = %d （累计解释 %.2f%%）\n', pcNum, cumexp(pcNum));

Xpca = score(:,1:pcNum);   % N x pcNum

%% 4. 构造模型（在 PCA 空间上拟合高斯）
nVars_pca = pcNum;
% 参数向量 p = [a; b(1..m); c(1..m)]
p0 = [mean(y); 0.01*ones(nVars_pca,1); mean(Xpca,1)'];
lb = [-Inf; zeros(nVars_pca,1); min(Xpca,[],1)'];
ub = [ Inf; ones(nVars_pca,1)*Inf;  max(Xpca,[],1)'];

% 正则化强度（可调）
lambda = 1e-0;   % 尝试 1e-4 ~ 1e0 看效果

% 残差函数（扩展，加入正则项）
resid_fun = @(p) [ gaussian_model(p, Xpca, nVars_pca) - y; ...
                   sqrt(lambda) * p(2:1+nVars_pca); ...
                   sqrt(lambda) * (p(2+nVars_pca:end)) ];

% 优化选项
if exist('lsqnonlin','file') == 2
    opts = optimoptions('lsqnonlin', 'Display', 'iter', 'MaxIterations', 3000, 'FunctionTolerance', 1e-12);
    try
        p_opt = lsqnonlin(resid_fun, p0, [], [], opts);
    catch ME
        warning('lsqnonlin 失败：%s\n尝试 lsqcurvefit（若可用）或回退到普通最小二乘。', ME.message);
        % 回退到 lsqcurvefit if available
        if exist('lsqcurvefit','file') == 2
            model_wrapper = @(p,Xdata) gaussian_model(p,Xpca,nVars_pca); % Xpca captured
            p_opt = lsqcurvefit(model_wrapper, p0, Xpca, y, lb, ub, optimoptions('lsqcurvefit','Display','iter'));
        else
            error('没有可用的非线性最小二乘求解器，请安装 Optimization Toolbox 或者让我发 Adam 版本。');
        end
    end
else
    error('lsqnonlin 不可用。请安装 Optimization Toolbox，或请求不依赖工具箱的版本。');
end

%% 5. 结果回代并评估（注意：b,c 在 PCA 空间）
a_opt = p_opt(1);
b_opt = p_opt(2:1+nVars_pca);
c_opt = p_opt(2+nVars_pca:end);

y_pred_pca = gaussian_model(p_opt, Xpca, nVars_pca);
residual = y_pred_pca - y;

fprintf('\nPCA+正则 lsqnonlin 结果：\n');
fprintf('  a = %.6g\n', a_opt);
fprintf('残差统计： mean=%.4f, max=%.4f, min=%.4f, std=%.4f\n', mean(residual), max(residual), min(residual), std(residual));

% 图像诊断
figure;
subplot(1,2,1);
plot(y, y_pred_pca, 'bo'); hold on; plot([min(y),max(y)],[min(y),max(y)],'r--');
xlabel('实际 y'); ylabel('预测 y'); title('PCA+正则：预测 vs 实际'); axis equal; grid on;
subplot(1,2,2);
histogram(residual, 30); xlabel('残差'); title('残差分布'); grid on;

% 显示 top5 残差样本以便核查
[~, idxs] = sort(abs(residual),'descend');
topk = min(10, nSamples);
fprintf('top %d 绝对残差样本（index, residual）：\n', topk);
disp([idxs(1:topk), residual(idxs(1:topk))]);

%% 6. 选项：把 PCA 空间的 c,b 转回原始 X 空间（若你确实需要解释）
% 说明：c 是 PCA 坐标上的位置；要把它映射回原始特征空间：
% c_orig = (c_pca * coeff(:,1:pcNum)') .* sigmaX + muX  （需要小心矩阵维度）
% 这里不自动回算——若你需要，我可以给出精确的恢复步骤（并考虑标准化因子）。





%% ---------- 局部函数：高斯模型（在当前文件） ------------
function ypred = gaussian_model(p, Xdata, nVars_local)
    a = p(1);
    b = p(2:1+nVars_local);
    c = p(2+nVars_local:end);
    diffs = (Xdata - repmat(c', size(Xdata,1), 1)).^2;
    expo  = exp( - diffs * b );
    ypred = a * expo;

end