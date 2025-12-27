function newGradient()
% ROBUST_GAUSS_FIT  稳健高斯拟合示例（主函数）
% 将 gaussian_model 单独放在 gaussian_model.m

clc; clearvars -except robust_gauss_fit; close all;

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
D = D_all(:, 1:(size(D_all,2)-1));   % 去掉最后一列避免虚拟变量陷阱
X = [X0, D];

[nSamples, nVars] = size(X);

%% ------------------ 3. 初始值 / 边界 ------------------
p0 = [mean(y); 0.01*ones(nVars,1); mean(X,1)'];  % [a; b(1..n); c(1..n)]
lb = [-Inf; zeros(nVars,1); min(X,[],1)'];
ub = [ Inf; ones(nVars,1)*Inf;  max(X,[],1)'];

%% ------------------ 4. 优先：nlinfit + Robust (bisquare) ------------------
do_fallback = false;
if exist('nlinfit','file') == 2
    fprintf('检测到 nlinfit，尝试 robust nlinfit (bisquare) 进行拟合...\n');
    modelfun = @(p, Xdata) gaussian_model(p, Xdata, nVars);  % 通过外部文件调用
    options = statset('nlinfit');
    options.Display = 'iter';
    options.MaxIter = 200;
    options.RobustWgtFun = 'bisquare';
    try
        [p_robust, resid_robust, J, CovB, MSE, stats] = nlinfit(X, y, modelfun, p0, options);
    catch ME
        warning('nlinfit 运行失败：%s\n将尝试 IRLS 回退方案。', ME.message);
        do_fallback = true;
    end
    if exist('p_robust','var')
        p_opt = p_robust;
        y_pred = gaussian_model(p_opt, X, nVars);
        residual = y_pred - y;
        fprintf('Robust nlinfit 完成。\n');
        if exist('stats','var') && isfield(stats,'w')
            weights = stats.w;
        else
            weights = ones(nSamples,1);
        end
        fprintf('平均误差 = %.4f， 最大误差 = %.4f， 最小误差 = %.4f\n', mean(residual), max(residual), min(residual));
        [ws_sorted, idxw] = sort(weights);
        kshow = min(10, nSamples);
        fprintf('权重最小的 %d 个样本 (index, weight, residual):\n', kshow);
        disp([idxw(1:kshow), ws_sorted(1:kshow), residual(idxw(1:kshow))]);
    end
else
    do_fallback = true;
end

%% ------------------ 5. 回退方案：IRLS + lsqnonlin (若没有 nlinfit 或失败) ------------------
if do_fallback
    fprintf('使用回退方案：IRLS (bisquare) + lsqnonlin。\n');
    if exist('lsqnonlin','file') ~= 2
        error(['既没有 nlinfit 也没有 lsqnonlin，无法进行稳健拟合。', ...
               ' 请安装 Statistics Toolbox 或 Optimization Toolbox，或要求我提供不依赖工具箱的版本。']);
    end
    maxIters = 12;
    tolP = 1e-6;
    p_current = p0;
    opts_lsq = optimoptions('lsqnonlin','Display','off','MaxIterations',1000,'FunctionTolerance',1e-12);
    try
        p_current = lsqnonlin(@(pt) gaussian_model(pt,X,nVars) - y, p_current, lb, ub, opts_lsq);
    catch
        warning('初始 lsqnonlin 拟合失败，使用初始 p0 继续 IRLS。');
        p_current = p0;
    end
    for it = 1:maxIters
        yhat = gaussian_model(p_current, X, nVars);
        r = yhat - y;
        s = 1.4826 * median(abs(r - median(r)));
        if s <= 0
            s = std(r); if s <= 0, s = 1; end
        end
        c = 4.6851 * s;
        absr = abs(r);
        w = zeros(size(r));
        mask = absr <= c;
        w(mask) = (1 - (r(mask)./c).^2).^2;
        w = max(w, 1e-6);
        weighted_resid = @(pt) (sqrt(w) .* (gaussian_model(pt, X, nVars) - y));
        try
            p_next = lsqnonlin(weighted_resid, p_current, lb, ub, opts_lsq);
        catch ME
            warning('IRLS 第 %d 次 lsqnonlin 失败：%s。跳出 IRLS。', it, ME.message);
            break;
        end
        if norm(p_next - p_current) < tolP * (1 + norm(p_current))
            p_current = p_next;
            fprintf('IRLS 收敛于第 %d 次迭代。\n', it);
            break;
        end
        p_current = p_next;
        if it == maxIters
            fprintf('IRLS 达到最大迭代次数 %d。\n', maxIters);
        end
    end
    p_opt = p_current;
    y_pred = gaussian_model(p_opt, X, nVars);
    residual = y_pred - y;
    s = 1.4826 * median(abs(residual - median(residual)));
    if s <= 0, s = std(residual); if s <= 0, s = 1; end
    c = 4.6851 * s;
    absr = abs(residual);
    w_final = zeros(size(residual));
    mask = absr <= c;
    w_final(mask) = (1 - (residual(mask)./c).^2).^2;
    w_final = max(w_final, 1e-6);
    weights = w_final;
    fprintf('IRLS 最终结果：平均误差 = %.4f， 最大误差 = %.4f， 最小误差 = %.4f\n', mean(residual), max(residual), min(residual));
    [ws_sorted, idxw] = sort(weights);
    kshow = min(10, nSamples);
    fprintf('权重最小的 %d 个样本 (index, weight, residual):\n', kshow);
    disp([idxw(1:kshow), ws_sorted(1:kshow), residual(idxw(1:kshow))]);
end

%% ------------------ 6. 输出最终参数 ------------------
a_opt = p_opt(1);
b_opt = p_opt(2:1+nVars);
c_opt = p_opt(2+nVars:end);

fprintf('\n最终参数（近似或稳健估计）：\n');
fprintf('  a = %.8g\n', a_opt);
for i = 1:nVars
    fprintf('  b(%d) = %.8g, c(%d) = %.8g\n', i, b_opt(i), i, c_opt(i));
end

fprintf('\n残差统计：\n');
fprintf('  mean = %.6g\n  median = %.6g\n  std = %.6g\n  max = %.6g\n  min = %.6g\n', mean(residual), median(residual), std(residual), max(residual), min(residual));

%% ------------------ 7. 画图诊断 ------------------
figure('Name','稳健拟合诊断','Position',[100 100 1000 600]);
subplot(2,2,1);
plot(y, y_pred, 'bo'); hold on;
plot([min(y),max(y)], [min(y),max(y)], 'r--','LineWidth',1.2);
xlabel('实际 y'); ylabel('预测 y'); title('预测 vs 实际'); grid on; axis equal;
subplot(2,2,2);
histogram(residual, 30);
xlabel('残差'); ylabel('频数'); title('残差分布'); grid on;
subplot(2,2,3);
scatter(1:nSamples, residual, 30, weights, 'filled');
colorbar; xlabel('样本索引'); ylabel('残差'); title('残差与稳健权重 (颜色表示)'); grid on;
subplot(2,2,4);
[~, idxs] = sort(abs(residual),'descend');
topk = min(20, nSamples);
barh(abs(residual(idxs(1:topk))));
set(gca,'YTickLabel', idxs(1:topk));
xlabel('|残差|'); title(sprintf('Top %d 绝对残差（索引）', topk)); grid on;

%% ------------------ 8. 导出被 downweight 的点（便于人工检查） ------------------
kdown = min(20, nSamples);
[~, idxw2] = sort(weights);
lowIndices = idxw2(1:kdown);
fprintf('\n建议人工检查的被 downweight 样本索引（前 %d）：\n', kdown);
disp(lowIndices');

end  % 结束主函数 robust_gauss_fit



