%% print_fitted_gauss.m
% 把当前拟合得到的 p 参数打印为可读函数与 LaTeX
% 运行前请确保工作区里有 p_opt 或 p_robust 或 beta（nlinfit 返回）变量。
% 若你用 PCA 版拟合，确保 coeff 或 Xpca/pcNum 也在工作区，脚本会提示。

% 1) 找到参数向量
if exist('p_opt','var')
    p = p_opt;
elseif exist('p_robust','var')
    p = p_robust;
elseif exist('beta','var')   % nlinfit 有时返回名为 beta
    p = beta;
elseif exist('p_clean','var') % 其他变量名备选
    p = p_clean;
else
    error('未找到参数向量 p_opt / p_robust / beta 等。请先在工作区运行拟合并保留参数变量。');
end

% 2) 判定是否是 PCA 空间（优先提示）
isPCA = exist('coeff','var') && exist('Xpca','var');

% 3) 变量名构造（优先从表格里读取真实变量名）
varNames = {};
try
    % 如果存在 T1,T2, T3, T_brand，按你原流程构造名字
    if exist('T1','var') && exist('T2','var') && exist('T_brand','var')
        names1 = T1.Properties.VariableNames;
        names2 = T2.Properties.VariableNames;
        % 时间变量名在 T3 中
        if exist('T3','var') && ismember('时间量化', T3.Properties.VariableNames)
            names_time = {'时间量化'};
        else
            names_time = {'time'};
        end
        % 品牌类名
        brandCats = categories(categorical(T_brand.Brand));
        % 你之前用 dummyvar 并去掉最后一列，所以品牌列名为 brandCat(1:end-1)
        brandCols = cellfun(@(s) ['Brand_' char(s)], brandCats, 'UniformOutput', false);
        if numel(brandCols) > 0
            brandCols = brandCols(1:end-1); % 与你做 D_all(:,1:end-1) 对应
        end
        varNames = [names1, names2, names_time, brandCols];
    end
catch
    varNames = {};
end

% 如果没获得 varNames，则用通用 x1..xn
nParams = numel(p);
% infer nVars from p: p = [a; b(1..n); c(1..n)] => n = (len-1)/2
if mod(nParams-1,2)~=0
    % 可能是 PCA 版本或不同格式
    % fallback: use generic names x1..xn where n = floor((len-1)/2)
    nVars_inferred = floor((nParams-1)/2);
else
    nVars_inferred = (nParams-1)/2;
end
if isempty(varNames) || numel(varNames) < nVars_inferred
    varNames = arrayfun(@(k) sprintf('x%d',k), 1:nVars_inferred, 'UniformOutput', false);
end
varNames = varNames(1:nVars_inferred);

% 4) 数字格式（可改）
fmt = '%.6g';  % 用 6 位有效数字，改成 '%.8g' 或 '%.4f' 视需求

% 5) 构造函数字符串（紧凑形式 & 展开形式）
a = p(1);
b = p(2:1+nVars_inferred);
c = p(2+nVars_inferred:end);

% 紧凑式
terms = cell(nVars_inferred,1);
for i=1:nVars_inferred
    terms{i} = sprintf('%s*(%s - %s)^2', num2str(b(i),fmt), varNames{i}, num2str(c(i),fmt));
end
inner = strjoin(terms, ' + ');
compact = sprintf('f(x) = %s * exp( - ( %s ) )', num2str(a,fmt), inner);

% 展开友好式（把每项显式写出来）
expanded_terms = cell(nVars_inferred,1);
for i=1:nVars_inferred
    expanded_terms{i} = sprintf('%s * (%s - %s)^2', num2str(b(i),fmt), varNames{i}, num2str(c(i),fmt));
end
expanded = sprintf('f(x) = %s * exp( - ( %s ) )', num2str(a,fmt), strjoin(expanded_terms,' + '));

% 6) 打印到命令窗口
fprintf('\n===== 拟合函数（紧凑） =====\n%s\n', compact);
fprintf('\n===== 拟合函数（展开） =====\n%s\n', expanded);

% 7) 生成更可读的多行表示（便于查看）
fprintf('\n===== 多行可读表示 =====\n');
fprintf('f(x) = %s * exp( - (\n', num2str(a,fmt));
for i=1:nVars_inferred
    fprintf('    %s * (%s - %s)^2%s\n', num2str(b(i),fmt), varNames{i}, num2str(c(i),fmt), ternary(i<nVars_inferred, ',', ''));
end
fprintf(') )\n\n');

% 8) 写出到文本文件
txtfile = 'fitted_gauss_function.txt';
fid = fopen(txtfile,'w');
if fid>0
    fprintf(fid, '拟合时间: %s\n\n', datestr(now));
    fprintf(fid, '紧凑式:\n%s\n\n', compact);
    fprintf(fid, '展开式:\n%s\n\n', expanded);
    fprintf(fid, '多行可读:\n');
    fprintf(fid, 'f(x) = %s * exp( - (\n', num2str(a,fmt));
    for i=1:nVars_inferred
        fprintf(fid, '    %s * (%s - %s)^2%s\n', num2str(b(i),fmt), varNames{i}, num2str(c(i),fmt), ternary(i<nVars_inferred, ',', ''));
    end
    fprintf(fid, ') )\n');
    fclose(fid);
    fprintf('已保存纯文本到: %s\n', fullfile(pwd, txtfile));
else
    warning('无法写入 %s', txtfile);
end

% 9) 生成 LaTeX 版本并保存
texfile = 'fitted_gauss_function.tex';
fid = fopen(texfile,'w');
if fid>0
    fprintf(fid, '%% LaTeX 表达式（自动生成）\n');
    fprintf(fid, '%% 使用 \\( ... \\) 嵌入行内数学或 \\[ ... \\] 行间数学\n\n');
    fprintf(fid, '\\[ f(x) = %s \\exp\\Big( - %s \\Big) \\]\n\n', num2str(a,fmt), latex_inner(b,c,varNames,fmt));
    fclose(fid);
    fprintf('已保存 LaTeX 到: %s\n', fullfile(pwd, texfile));
else
    warning('无法写入 %s', texfile);
end

% 10) PCA 情况提示
if isPCA
    fprintf('\n注意：检测到 PCA 变量（coeff / Xpca 存在），当前函数为 PCA 空间上的表示，\n');
    fprintf('若需要把 c 参数转换回原始特征空间请告知，我会给出转换代码。\n');
end

% ----------------- 辅助本地函数 -----------------
function out = latex_inner(bvec,cvec,names,fmtLocal)
    parts = cell(numel(bvec),1);
    for kk = 1:numel(bvec)
        bi = num2str(bvec(kk),fmtLocal);
        ci = num2str(cvec(kk),fmtLocal);
        % latex variable name (escape underscores)
        vn = names{kk};
        vn = strrep(vn,'_','\_');
        parts{kk} = sprintf('%s ( %s - %s )^2', bi, vn, ci);
    end
    out = strjoin(parts,' + ');
end

function r = ternary(cond, a, b)
    if cond, r = a; else r = b; end
end
