% function mixed_reg_gauss_fit()

%梯度下降的最终答案

%% ---------- 0. Settings设置 ----------
rng(0);                 % for reproducibility
usePCA = true;          % whether to fit Gaussian on PCA components (recommended)
explained_target = 95;  % PCA cumulative % to keep (if usePCA)
nStarts = 10;           % multi-start count (increase for better robustness)
lambda_b = 1e-3;        % regularization for b
lambda_c = 1e-3;        % regularization for c
lambda_w = 1e-3;        % regularization for linear weights
maxLSQIter = 2500;      % lsqnonlin max iter
showPlots = true;       % show diagnostic plots
saveResults = true;     % save results to MAT & text

%% ---------- 1. Load & prepare data (adapt to your sheet/col names)载入表格数据 ----------
fprintf('Loading data...\n');
filename = 'data3.xlsx';
T1       = readtable(filename, 'Sheet', '性能',     'VariableNamingRule', 'preserve');
T2       = readtable(filename, 'Sheet', '周边配置', 'VariableNamingRule', 'preserve');
T3       = readtable(filename, 'Sheet', '价格',     'VariableNamingRule', 'preserve');
T_brand  = readtable(filename, 'Sheet', '品牌',     'VariableNamingRule', 'preserve');

% pick same columns you used before
T3_vars = T3(:, {'时间量化','价格差值'});
X_time  = T3_vars.('时间量化');
y       = T3_vars.('价格差值');

T1 = T1(:, {'SoC制程量化','跑分量化'});
T2 = T2(:, {'存储量化','RAM量化','屏幕尺寸量化','分辨率量化', ...
            '刷新率量化','主摄量化','电池容量量化','充电功率量化'});
X0 = [T1{:,:}, T2{:,:}, X_time];

% brand dummies (drop last column to avoid dummy trap)
brand = categorical(T_brand.Brand);
D_all = dummyvar(brand);
if size(D_all,2) >= 1
    D = D_all(:,1:(size(D_all,2)-1));
else
    D = [];
end

X_full = [X0, D];         % full features
[nSamples, nVarsFull] = size(X_full);
fprintf('Data: %d samples, %d raw features\n', nSamples, nVarsFull);

% drop constant columns
colStd = std(X_full,0,1);
constCols = find(colStd < 1e-8);
if ~isempty(constCols)
    fprintf('Dropping %d near-constant columns (indices):\n', numel(constCols));
    disp(constCols);
    X_full(:,constCols) = [];
    nVarsFull = size(X_full,2);
end

%% ---------- 2. Standardize features 标准化----------
muX = mean(X_full,1);
sigmaX = std(X_full,0,1); sigmaX(sigmaX==0) = 1;
Xn = (X_full - muX) ./ sigmaX;  % standardized features used for linear part

%% ---------- 3. PCA for Gaussian part (optional) 给高斯函数降维----------
if usePCA
    fprintf('Running PCA on standardized features...\n');
    [coeff, score, ~, ~, explained] = pca(Xn);
    cumexp = cumsum(explained);
    pcNum = find(cumexp >= explained_target,1,'first');
    if isempty(pcNum)
        pcNum = min(10, size(Xn,2));
    end
    pcNum = max(pcNum, min(10, size(Xn,2)));  % ensure at least 10 PCs (or <= dims)
    fprintf('Keeping %d PCs (%.2f%% variance explained)\n', pcNum, cumexp(pcNum));
    Z = score(:,1:pcNum);   % PCA coordinates used for Gaussian
    
    % 保存 PCA 变换参数（均值、标准差、投影矩阵）
    mu_X = mean(X);         % 原始 X 每列均值
    sigma_X = std(X);       % 原始 X 每列标准差
    coeff_PCA = coeff(:, 1:pcNum);  % 只保留用到的主成分

    save('pca_transform.mat', 'mu_X', 'sigma_X', 'coeff_PCA');

    % 定义一个将原始 x 转换到 z 的函数
    x2z = @(xrow) ((xrow - mu_X) ./ sigma_X) * coeff_PCA;

else
    fprintf('Not using PCA: Gaussian will be fit on standardized original features.\n');
    Z = Xn;
    pcNum = size(Z,2);
end

m_gauss = pcNum;          % gaussian dimension
k_lin = size(Xn,2);       % linear features count

%% ---------- 4. parameter vector layout 残差函数----------
% p = [a; b(1..m); c(1..m); w(1..k); bias]
nParams = 1 + 2*m_gauss + k_lin + 1;
fprintf('Parameter length = %d (a + 2*m + k + bias)\n', nParams);

%% ---------- 5. residual function (with L2 reg) ----------
residual_fn = @(p) mixed_residual(p, Z, Xn, y, m_gauss, k_lin, lambda_b, lambda_c, lambda_w);

%% ---------- 6. optimizer availability check 检查优化器----------
if exist('lsqnonlin','file') ~= 2
    error(['lsqnonlin not available. Please install Optimization Toolbox, ' ...
           'or request an alternative (Adam-based) implementation).']);
end

opts = optimoptions('lsqnonlin', 'Display', 'iter-detailed', 'MaxIterations', maxLSQIter, ...
    'FunctionTolerance',1e-12, 'StepTolerance',1e-12);

%% ---------- 7. multi-start lsqnonlin ----------
fprintf('Multi-start lsqnonlin with %d starts...\n', nStarts);
bestResnorm = Inf;
bestP = [];
resnormHistory = nan(nStarts,1);
timeStart = tic;

for s = 1:nStarts
    % build a reasonable p0
    p0 = zeros(nParams,1);
    p0(1) = mean(y) * (1 + 0.05*randn);                          % a
    p0(2:1+m_gauss) = 0.01 * abs(1 + 0.2*randn(m_gauss,1));      % b small positive
    p0(2+m_gauss:1+2*m_gauss) = mean(Z,1)' + 0.1*randn(m_gauss,1); % c around mean of Z
    idxw = 2 + 2*m_gauss;
    p0(idxw:idxw + k_lin - 1) = 0.01 * randn(k_lin,1);           % w small
    p0(end) = 0;                                                % bias initial
    
    try
        [p_sol, resnorm, residuals, exitflag, output] = lsqnonlin(residual_fn, p0, [], [], opts);
    catch ME
        warning('Start %d failed: %s', s, ME.message);
        resnorm = Inf;
    end
    
    resnormHistory(s) = resnorm;
    fprintf('Start %d finished: resnorm = %.6g\n', s, resnorm);
    if isfinite(resnorm) && resnorm < bestResnorm
        bestResnorm = resnorm;
        bestP = p_sol;
        bestResiduals = residuals;
        bestOutput = output;
        bestExitflag = exitflag;
        bestStartIndex = s;
    end
end

totalTime = toc(timeStart);
fprintf('Multi-start done in %.1f s. Best resnorm = %.6g (start %d)\n', totalTime, bestResnorm, bestStartIndex);

if isempty(bestP)
    error('All starts failed. No solution found.');
end

%% ---------- 8. unpack best solution & evaluate ----------
p = bestP;
[a_opt, b_opt, c_opt, w_opt, bias_opt] = unpack_p(p, m_gauss, k_lin);

% Predict in Z-space and back to original predictions
G = a_opt * exp(- sum( (Z - c_opt').^2 .* (ones(size(Z,1),1)*b_opt'), 2));
L = Xn * w_opt + bias_opt;
yhat = G + L;
residual = yhat - y;

fprintf('Result statistics: mean residual = %.4f, max = %.4f, min = %.4f, std = %.4f\n', ...
    mean(residual), max(residual), min(residual), std(residual));

%% ---------- 9. Diagnostics & plots ----------
if showPlots
    fprintf('Generating diagnostic plots...\n');
    figure('Name','Mixed Model Diagnostics','Units','normalized','Position',[0.05 0.05 0.9 0.8]);
    
    subplot(2,3,1);
    scatter(y, yhat, 15, 'filled'); hold on;
    plot([min(y),max(y)],[min(y),max(y)], 'r--','LineWidth',1.2);
    xlabel('Actual y'); ylabel('Predicted y');
    title('Predicted vs Actual'); grid on; axis equal;
    
    subplot(2,3,2);
    histogram(residual, 40);
    xlabel('Residual'); title('Residual distribution'); grid on;
    
    subplot(2,3,3);
    scatter(1:length(residual), residual, 18, abs(residual), 'filled');
    colorbar; xlabel('Sample index'); ylabel('Residual'); title('Residuals (color=|res|)'); grid on;
    
    subplot(2,3,4);
    [~, idxs] = sort(abs(residual),'descend');
    topk = min(20, length(residual));
    barh(abs(residual(idxs(1:topk))));
    set(gca,'YTickLabel', idxs(1:topk)); xlabel('|Residual|'); title(sprintf('Top %d absolute residuals', topk));
    grid on;
    
    % plot linear weights (map back to original feature names if possible)
    subplot(2,3,5);
    % build feature names
    varNames = build_varnames(T1, T2, T_brand);
    if numel(varNames) ~= k_lin
        % fallback generic names
        varNames = arrayfun(@(i) sprintf('x%d',i), 1:k_lin, 'UniformOutput', false);
    end
    [w_sorted, idxw] = sort(abs(w_opt),'descend');
    top_w = min(20, k_lin);
    barh(w_opt(idxw(1:top_w)));
    set(gca,'YTickLabel', varNames(idxw(1:top_w)));
    title('Top linear weights (signed)'); grid on;
    
    % map Gaussian center c back to original feature space for interpretation
    subplot(2,3,6);
    c_orig = map_c_to_original(c_opt', coeff, usePCA, muX, sigmaX);  % returns 1 x p vector
    [cabs, cidx] = sort(abs(c_orig),'descend');
    topc = min(20, numel(c_orig));
    barh(c_orig(cidx(1:topc)));
    set(gca,'YTickLabel', build_varnames(T1, T2, T_brand, cidx(1:topc)));
    title('Gaussian center (mapped back to original features)'); grid on;
end

%% ---------- 10. Save results & print fitted function ----------
if saveResults
    savename = sprintf('mixed_gauss_fit_result_%s.mat', datestr(now,'yyyymmdd_HHMMSS'));
    save(savename, 'p', 'a_opt','b_opt','c_opt','w_opt','bias_opt','residual','yhat','Z','Xn','muX','sigmaX','coeff');
    fprintf('Saved results to %s\n', savename);
    
    % Print simple human readable function
    print_mixed_function(a_opt, b_opt, c_opt, w_opt, bias_opt, varNames);
end

%% ---------- 11. show top residual rows for manual check ----------
[~, idxsAll] = sort(abs(residual),'descend');
topk = min(20, length(residual));
fprintf('\nTop %d absolute residual samples (index, residual):\n', topk);
disp([idxsAll(1:topk), residual(idxsAll(1:topk))]);


%% ===== 在 PCA 那段代码后面插入 =====
% 计算 dz/dx 矩阵 (pcNum × 原始特征数)
% 注意：Xn = (X - mu) ./ sigma，所以 dz/dx = coeff' * diag(1./sigma)
mu_X = mean(X);           % 每个特征的均值 (原始 X)
sigma_X = std(X, 0, 1);   % 每个特征的标准差 (原始 X)

dzdx = bsxfun(@rdivide, coeff(:,1:pcNum)', sigma_X);  
% dzdx 的大小是 (pcNum × 原始特征数)
% dzdx(j,i) = ∂z_j / ∂x_i

% 保存下来，后面做链式法则用
save('dzdx_matrix.mat','dzdx','mu_X','sigma_X');
fprintf('dz/dx 矩阵已计算并保存。\n');

%顺带算出x对G的偏导数值

load('dzdx_matrix.mat','dzdx','mu_X','sigma_X');
% coeff_PCA，b_opt，c_opt，a_opt 已经在你拟合后变量里有了

new_x = X(28,:);  % 举例某一行样本
[G_val, grad_G_x] = compute_G_and_grad(new_x, dzdx, mu_X, sigma_X, coeff, b_opt, c_opt, a_opt);

fprintf('G(z) = %.4f\n', G_val);
fprintf('dG/dx = \n');
disp(grad_G_x);

fprintf('\n下面附带计算p_linear.\n');

pcNum = length(b_opt);  % 主成分数量

a_opt = p(1);
b_opt = p(2 : 1 + pcNum);
c_opt = p(2 + pcNum : 1 + 2*pcNum);
p_opt_linear = p(2 + 2*pcNum : end);  % 剩下的就是线性权重了

fprintf('线性权重数量 = %d\n', length(p_opt_linear));
disp(p_opt_linear');

intercept = p_opt_linear(1);       % 截距
beta = p_opt_linear(2:end);        % 其余17个线性权重
L_pred = X * beta + intercept;

fprintf('\nDone.\n');%主程序终结！！！！！！！！！！！！！！！！！


  % end main function 现在没必要了


%% ================= Helper functions =================

function r = mixed_residual(p, Zloc, Xlinloc, yloc, mloc, kloc, lb_b, lb_c, lb_w)
    % Unpack
    [a_loc,b_loc,c_loc,w_loc,bias_loc] = unpack_p(p, mloc, kloc);
    % Gaussian part
    G = a_loc * exp(- sum( (Zloc - c_loc').^2 .* (ones(size(Zloc,1),1)*b_loc'), 2));
    L = Xlinloc * w_loc + bias_loc;
    resid = (G + L) - yloc;
    % regularization as extra residuals
    r = [resid; sqrt(lb_b) * b_loc; sqrt(lb_c) * c_loc; sqrt(lb_w) * w_loc];
end

function [a_loc,b_loc,c_loc,w_loc,bias_loc] = unpack_p(pvec, mloc, kloc)
    a_loc = pvec(1);
    b_loc = pvec(2:1+mloc);
    c_loc = pvec(2+mloc:1+2*mloc);
    idx2 = 2 + 2*mloc;
    w_loc = pvec(idx2:idx2+kloc-1);
    bias_loc = pvec(end);
end

function names = build_varnames(T1, T2, T_brand, idxSub)
% BUILD_VARNAMES Robustly assemble variable names similar to pipeline
% Usage:
%   names = build_varnames(T1,T2,T_brand)
%   names = build_varnames(T1,T2,T_brand, idxSub)  % returns subset by indices

    if nargin < 4
        useIdx = [];
    else
        useIdx = idxSub;
    end

    % safe extraction of variable names, always produce 1xN cell array of chars
    try
        names1 = cellstr(T1.Properties.VariableNames);
    catch
        names1 = {};
    end
    try
        names2 = cellstr(T2.Properties.VariableNames);
    catch
        names2 = {};
    end

    nameTime = {'时间量化'};   % keep as 1x1 cell

    % brand columns: Brand_<cat>, drop last to match dummyvar(...,1:end-1)
    try
        brandCats = categories(categorical(T_brand.Brand));
        brandCols = cellfun(@(s) ['Brand_' char(s)], brandCats, 'UniformOutput', false);
        if numel(brandCols) > 0
            brandCols = brandCols(1:end-1);
        end
    catch
        brandCols = {};
    end

    % ensure all are row cell vectors
    names1   = reshape(names1, 1, []);
    names2   = reshape(names2, 1, []);
    nameTime = reshape(nameTime, 1, []);
    brandCols= reshape(brandCols,1,[]);

    % concatenate safely
    allNames = [names1, names2, nameTime, brandCols];

    % return subset if requested (guard bounds)
    if isempty(useIdx)
        names = allNames;
    else
        useIdx = useIdx(:)';
        useIdx = useIdx(useIdx >= 1 & useIdx <= numel(allNames)); % valid indices
        names = allNames(useIdx);
    end
end


function c_orig = map_c_to_original(c_pca, coeff, usedPCA, muX, sigmaX)
    % Map Gaussian center c (in PCA space) back to original feature space
    % c_pca: 1 x m  (row)
    % coeff: p x p from pca (if used)
    if ~usedPCA
        % if not PCA, c_pca already corresponds to standardized original features
        c_std = c_pca;  % 1 x p
    else
        % Xn_row = c_pca * coeff(:,1:m)'  -> gives standardized original features
        c_std = c_pca * coeff(:,1:size(c_pca,2))';
    end
    % map back to original scale
    c_orig = c_std .* sigmaX + muX;  % 1 x p
end

function print_mixed_function(a, b, c, w, bias, varNames)
    n_gauss = numel(b);
    k_lin = numel(w);
    fmt = '%.6g';
    fprintf('\n=== Fitted mixed model (human-readable) ===\n');
    % Gaussian part (print using z_j or feature names if available)
    fprintf('G(z) = %s * exp( - (', num2str(a,fmt));
    for j=1:n_gauss
        if j>1, fprintf(' + '); end
        fprintf('%s*(z%d - %s)^2', num2str(b(j),fmt), j, num2str(c(j),fmt));
    end
    fprintf(') )\n');
    % Linear part (map to feature names)
    fprintf('Linear part: L(x) = ');
    for j=1:min(k_lin, numel(varNames))
        if j>1, fprintf(' + '); end
        fprintf('%s * %s', num2str(w(j),fmt), varNames{j});
    end
    if k_lin > numel(varNames)
        fprintf(' + ... (%d more weights)', k_lin - numel(varNames));
    end
    fprintf(' + bias(%s)\n', num2str(bias,fmt));
    fprintf('Full model: yhat = G(z) + L(x)\n\n');
end

% 输入: new_x (1×nFeatures 原始变量向量)
% 你需要事先加载 dzdx, mu_X, sigma_X, b_opt, c_opt, a_opt

function [G_val, grad_G_x] = compute_G_and_grad(new_x, dzdx, mu_X, sigma_X, coeff_PCA, b_opt, c_opt, a_opt)
    % 标准化输入
    x_std = (new_x - mu_X) ./ sigma_X;        % 1×nFeatures
    
    % 计算 z
    z = x_std * coeff_PCA(:,1:length(b_opt)); % 1×pcNum，注意b_opt长度就是pcNum
    
    % 计算 G(z)
    diffs = (z - c_opt').^2;                   % 1×pcNum
    exp_term = exp(- sum(b_opt' .* diffs));
    G_val = a_opt * exp_term;
    
    % 对 z 求偏导 dG/dz
    grad_G_z = -2 * G_val * (b_opt' .* (z - c_opt'));  % 1×pcNum
    
    % 链式法则计算 dG/dx = dG/dz * dz/dx
    grad_G_x = grad_G_z * dzdx;                   % 1×nFeatures
end

