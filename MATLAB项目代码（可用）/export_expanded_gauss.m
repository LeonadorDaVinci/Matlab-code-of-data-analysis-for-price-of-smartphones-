% export_expanded_gauss.m
% 运行前请确认：当前文件夹内有 mixed_gauss_fit_result_*.mat（mixed_reg_gauss_fit 保存的）
% 运行后会生成 predict_mixed_gauss_expanded.m（接受原始 x，返回 yhat），并保存展开表达式文本。

files = dir('mixed_gauss_fit_result_*.mat');
if isempty(files)
    error('未找到 mixed_gauss_fit_result_*.mat，请先运行 mixed_reg_gauss_fit 并保存结果。');
end
[~, idx] = max([files.datenum]); fname = files(idx).name;
S = load(fname);

% 读取必要变量（尽量容错）
assert(isfield(S,'a_opt') && isfield(S,'b_opt') && isfield(S,'c_opt'), '结果文件里缺少 a_opt/b_opt/c_opt');
a = S.a_opt(:);
b = S.b_opt(:);    % m x 1
c = S.c_opt(:);    % m x 1
if isfield(S,'w_opt'), w = S.w_opt(:); else w = zeros(size(S.muX(:))); end
if isfield(S,'bias_opt'), bias = S.bias_opt; else bias = 0; end
muX = S.muX(:)'; sigmaX = S.sigmaX(:)';    % row
if isfield(S,'coeff')
    coeff = S.coeff;
else
    coeff = [];
end
p = numel(muX);  % 原始特征数
m = numel(b);

% 计算 Q, q, const （以标准化特征 x_std 为变量）
if ~isempty(coeff)
    V = coeff(:,1:m);     % p x m
    B = diag(b);          % m x m
    Q = V * B * V';       % p x p symmetric
    qvec = (V * (B * c)) ;% p x 1  (note: V*(B*c) is p x 1)
    constTerm = c' * (B * c); % scalar
else
    % 没有 PCA 情况： b,j 对应原始标准化特征直接
    Q = diag(b);          % p x p
    qvec = b .* c;
    constTerm = c' .* b' * c; % careful but if b and c are p-vectors
end

% 保存 Q, q, const 到 mat 便于检查
save('expanded_gauss_params.mat','Q','qvec','constTerm','a','w','bias','muX','sigmaX','fname');

% 生成预测函数文件
fid = fopen('predict_mixed_gauss_expanded.m','w');
fprintf(fid, 'function yhat = predict_mixed_gauss_expanded(x_orig)\n');
fprintf(fid, '%% predict_mixed_gauss_expanded: 使用原始特征直接预测（自动生成）\n');
fprintf(fid, '%% x_orig: N x p（p=%d），列须按 mixed_reg_gauss_fit 中的顺序\n\n', p);
fprintf(fid, 'a = %s;\n', mat2str(a,15));
fprintf(fid, 'Q = %s;\n', mat2str(Q,15));
fprintf(fid, 'qvec = %s;\n', mat2str(qvec(:).',15));
fprintf(fid, 'constTerm = %s;\n', num2str(constTerm,15));
fprintf(fid, 'w = %s;\n', mat2str(w(:).',15));
fprintf(fid, 'bias = %s;\n', num2str(bias,15));
fprintf(fid, 'muX = %s; sigmaX = %s;\n\n', mat2str(muX,15), mat2str(sigmaX,15));
fprintf(fid, 'if size(x_orig,2) == 1, x_orig = x_orig''; end\n');
fprintf(fid, 'Xn = (x_orig - muX) ./ sigmaX; %% standardize\n');
fprintf(fid, 'G = a * exp( - ( sum( (Xn * Q) .* Xn, 2 ) - 2 * (Xn * qvec'' ) + constTerm ) );\n');
fprintf(fid, 'L = Xn * w(:) + bias;\n');
fprintf(fid, 'yhat = G + L;\n');
fprintf(fid, 'end\n');
fclose(fid);

% 生成可读文本（top few terms）
txt = fopen('expanded_gauss_readable.txt','w');
fprintf(txt,'Expanded Gaussian (in standardized x):\n');
fprintf(txt,'Exponent = x_std'' * Q * x_std - 2 * q'' * x_std + const\n\n');
fprintf(txt,'const = %g\n\n', constTerm);
fprintf(txt,'Top absolute entries in Q (include cross terms):\n');
% list top K entries of Q by abs
K = min(40, p*(p+1)/2);
% collect upper triangular entries
list = [];
for i=1:p
    for j=i:p
        list = [list; i j Q(i,j)]; %#ok<AGROW>
    end
end
[~,ord] = sort(abs(list(:,3)),'descend');
for k=1:min(K, size(list,1))
    i = list(ord(k),1); j = list(ord(k),2); val = list(ord(k),3);
    fprintf(txt,'Q(%d,%d) = %.6g\n', i, j, val);
end
fclose(txt);

fprintf('已生成 predict_mixed_gauss_expanded.m 和 expanded_gauss_readable.txt（并保存 expanded_gauss_params.mat）\n');

