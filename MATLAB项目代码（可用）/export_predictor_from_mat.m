% export_predictor_from_mat.m
% 在运行 mixed_reg_gauss_fit 后运行此脚本：会生成 predict_mixed_gauss.m
% 并在命令窗口打印可读表达式。

clearvars -except
fprintf('寻找最新的 mixed_gauss_fit_result_*.mat...\n');
files = dir('mixed_gauss_fit_result_*.mat');
if isempty(files)
    error('未找到 mixed_gauss_fit_result_*.mat。请先运行 mixed_reg_gauss_fit 并保存结果。');
end
% 选最新的一个
[~, idx] = max([files.datenum]);
fname = files(idx).name;
fprintf('载入文件: %s\n', fname);
S = load(fname);

% 期望变量名： p,a_opt,b_opt,c_opt,w_opt,bias_opt,muX,sigmaX,coeff (mixed_reg_gauss_fit 保存的)
% 尝试容错取值
if isfield(S,'p')
    p = S.p;
end
% 优先使用命名的变量
if isfield(S,'a_opt'); a = S.a_opt; end
if isfield(S,'b_opt'); b = S.b_opt; end
if isfield(S,'c_opt'); c = S.c_opt; end
if isfield(S,'w_opt'); w = S.w_opt; end
if isfield(S,'bias_opt'); bias = S.bias_opt; end
if isfield(S,'muX'); muX = S.muX; end
if isfield(S,'sigmaX'); sigmaX = S.sigmaX; end
if isfield(S,'coeff'); coeff = S.coeff; end
if isfield(S,'Z'); Z = S.Z; end
if exist('b','var') && isempty(b) && exist('p','var')
    % try infer from p
    np = numel(p);
    nvars = (np-1)/2;
    if mod(np-1,2)==0
        % ambiguous when linear w present; best-effort fallback not used here
    end
end

% 校验关键变量存在性
req = {'a','b','c','w','bias','muX','sigmaX'};
miss = {};
for k=1:numel(req)
    if ~exist(req{k}, 'var')
        miss{end+1} = req{k}; %#ok<SAGROW>
    end
end
if ~isempty(miss)
    warning('导出时未找到全部变量（%s）。我会尽可能继续，但若缺少 muX/sigmaX/coeff 则无法正确生成 predictor。', strjoin(miss,', '));
end

% infer dimensions
m_gauss = numel(b);
k_lin = numel(w);

% 生成 predict_mixed_gauss.m
outname = 'predict_mixed_gauss.m';
fid = fopen(outname,'w');
if fid == -1
    error('无法创建 %s，请检查写权限。', outname);
end

fprintf(fid, 'function yhat = predict_mixed_gauss(x_orig)\n');
fprintf(fid, '%% PREDICT_MIXED_GAUSS 预测函数（自动生成）\n');
fprintf(fid, '%% 输入 x_orig: N x p 原始特征（与 mixed_reg_gauss_fit 中的特征顺序一致）\n');
fprintf(fid, '%% 输出 yhat: N x 1 预测值\n\n');

% write numeric params with good precision
fprintf(fid, '%% ---- 固定参数（由拟合结果生成） ----\n');
fprintf(fid, 'a = %s;\n', mat2str(a,15));
fprintf(fid, 'b = %s;\n', mat2str(b(:).',15));   % row
fprintf(fid, 'c = %s;\n', mat2str(c(:).',15));
fprintf(fid, 'w = %s;\n', mat2str(w(:).',15));
fprintf(fid, 'bias = %s;\n\n', mat2str(bias,15));

% muX sigmaX
if exist('muX','var') && exist('sigmaX','var')
    fprintf(fid, 'muX = %s;\n', mat2str(muX(:).',15));
    fprintf(fid, 'sigmaX = %s;\n\n', mat2str(sigmaX(:).',15));
else
    fprintf(fid, 'muX = [];\nsigmaX = [];\n\n');
end

% coeff (PCA) if exists
if exist('coeff','var')
    fprintf(fid, 'coeff = %s;\n\n', mat2str(coeff,15));
    fprintf(fid, 'usePCA = true;\n\n');
else
    fprintf(fid, 'coeff = [];\n\n');
    fprintf(fid, 'usePCA = false;\n\n');
end

% function body: check input dims
fprintf(fid, 'if isempty(x_orig)\n    yhat = [];\n    return;\nend\n\n');
fprintf(fid, 'if size(x_orig,2) == 1\n    x_orig = x_orig''; %% ensure row vector\nend\n\n');

% standardize
fprintf(fid, '%% 标准化原始特征\nif ~isempty(muX)\n    Xn = (x_orig - muX) ./ sigmaX; %% N x p\nelse\n    Xn = x_orig; %% assume already standardized\nend\n\n');

% compute Z (PCA) or use standardized X directly
fprintf(fid, 'if usePCA\n    %% 将原始标准化特征映射到 PCA 空间（与训练时一致）\n    m_gauss = %d;\n    Z = (Xn * coeff(:,1:m_gauss)); %% N x m\nelse\n    Z = Xn; %% N x m\nend\n\n', m_gauss);

% compute Gauss and linear
fprintf(fid, '%% 计算 Gaussian 部分\nb = b(:)''; c = c(:)''; %% ensure row\nG = a * exp( - sum( (Z - c).^2 .* (ones(size(Z,1),1) * b), 2) );\n\n');
fprintf(fid, '%% 计算线性部分\nL = Xn * w(:) + bias;\n\n');
fprintf(fid, 'yhat = G + L;\nend\n');
fclose(fid);

fprintf('已生成 %s 。你可以通过调用 predict_mixed_gauss(x) 预测。\n', outname);

% 也在命令行打印人类可读的表达式
fmt = '%.6g';
fprintf('\n=== 人类可读表达式（Gaussian + Linear） ===\n');
fprintf('G(z) = %s * exp( - ', num2str(a,fmt));
for j=1:m_gauss
    if j>1, fprintf(' - '); end
    fprintf('%s*(z%d - %s)^2', num2str(b(j),fmt), j, num2str(c(j),fmt));
end
fprintf(' )\n');
fprintf('L(x) = ');
for j=1:min(10,k_lin)
    if j>1, fprintf(' + '); end
    fprintf('%s * x_%d', num2str(w(j),fmt), j);
end
if k_lin > 10, fprintf(' + ... (剩余 %d 个权重)\n', k_lin-10); else fprintf('\n'); end
fprintf('Full: yhat = G(z) + L(x)\n\n');

% also save textual formula file
txtf = 'fitted_mixed_function.txt';
fid2 = fopen(txtf,'w');
if fid2~=-1
    fprintf(fid2, 'Fitted mixed model from %s\n\n', fname);
    fprintf(fid2, 'G(z) = %s * exp( - (', num2str(a,15));
    for j=1:m_gauss
        if j>1, fprintf(fid2,' + '); end
        fprintf(fid2, '%s*(z%d - %s)^2', num2str(b(j),15), j, num2str(c(j),15));
    end
    fprintf(fid2, ') )\n\n');
    fprintf(fid2, 'L(x) linear weights (first 20):\n');
    for j=1:min(20,k_lin)
        fprintf(fid2, '%d: %s\n', j, num2str(w(j),15));
    end
    fclose(fid2);
    fprintf('并将可读表达式保存到 %s\n', txtf);
end

