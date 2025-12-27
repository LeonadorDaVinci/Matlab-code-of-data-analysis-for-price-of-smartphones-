function yhat = predict_mixed_gauss_expanded_fixed(Xnum, D)
% PREDICT_EXPLICIT_17  直接使用 11 个数值特征 + 6 个哑变量的预测函数（单文件、无外部调用）
%
% 输入:
%   Xnum - N x 11 数值特征，列顺序必须与训练时一致 (T1(2), T2(8), 时间量化)
%   D    - N x 6 哑变量（每行是 one-hot except baseline）, 若只有一列编号也支持（见下）
% 输出:
%   yhat - N x 1 预测值
%
% 使用方法：
%   Xnum = [...];                % N x 11
%   D = [...];                   % N x 6 或 N x 1 (品牌编号)
%   yhat = predict_explicit_17(Xnum, D);
%
% 重要：把下面 "===== 下面的常量请替换为你拟合得到的数值 =====" 区域的占位符替换为你拟合出的实数矩阵/向量
% 例如：把  muX = [REPLACE_WITH_muX]; 替换为 muX = [0.123, 0.456, ...];

%% --------- 1) 输入处理与简单检查 ----------
if nargin < 2, D = []; end
Xnum = double(Xnum);
[N, nnum] = size(Xnum);
if nnum ~= 11
    error('Xnum 必须是 N x 11（当前列数 = %d）。请确认列顺序。', nnum);
end

% 支持 D 传入方式：N x 6 或 N x 1（品牌编号）或 []（全部基准）
if isempty(D)
    D = zeros(N,6);   % 全部 baseline（all zeros）
elseif isvector(D) && size(D,1)==N && size(D,2)==1
    brandIndex = D(:);
    ndummy = 6;
    maxBrand = ndummy + 1;  % baseline index = 7
    Dtmp = zeros(N, ndummy);
    for i=1:N
        bi = brandIndex(i);
        if bi>0 && bi<maxBrand
            Dtmp(i,bi) = 1;
        end
    end
    D = Dtmp;
else
    D = double(D);
    if size(D,1) ~= N
        error('D 的行数必须与 Xnum 行数一致。');
    end
    if size(D,2) ~= 6
        error('D 必须是 N x 6 或 N x 1（品牌编号）。 当前列数 = %d', size(D,2));
    end
end

% 拼成完整原始输入（未经标准化）
Xfull = [Xnum, D];  % N x 17

%% ===== 下面的常量请替换为你拟合得到的数值（粘入具体数组/矩阵） =====
% 举例替换方式： 在 MATLAB 命令行运行 disp(mat2str(muX,15)) 然后把输出粘到下面

% （示例占位，用真实数值替换这些行）
muX     = [REPLACE_WITH_muX];        % 1 x 17 向量（训练时的均值）
sigmaX  = [REPLACE_WITH_sigmaX];     % 1 x 17 向量（训练时的标准差，不能包含 0）
Q       = [REPLACE_WITH_Q];          % 17 x 17 对称矩阵 （用于展开形式）
qvec    = [REPLACE_WITH_qvec];      % 17 x 1 列向量
constTerm = REPLACE_WITH_constTerm;  % 标量
a       = REPLACE_WITH_a;            % 标量
w_full  = [REPLACE_WITH_w_full];     % 17 x 1 线性权重（标准化特征的权重）
bias    = REPLACE_WITH_bias;         % 标量
% （如果你的拟合没有用展开 Q 形式，而是用 PCA 的 b/c/coeff，请把训练时的 muX/sigmaX/coeff 以及 b,c,a 替换进来并告诉我，我可以给对应版本）

%% --------- 2) 标准化 ----------
% 安全处理 sigmaX 中的 0
sigmaX(sigmaX==0) = 1;
Xn = (Xfull - muX) ./ sigmaX;   % N x 17

%% --------- 3) Gaussian 部分（展开二次型形式） ----------
% exponent = x_std' * Q * x_std - 2 * qvec' * x_std + constTerm
quad = sum((Xn * Q) .* Xn, 2);    % N x 1
lin_q = 2 * (Xn * qvec);          % N x 1
exponent = quad - lin_q + constTerm;
G = a * exp(-exponent);           % N x 1

%% --------- 4) 线性部分 ----------
if isempty(w_full)
    L = bias + zeros(N,1);
else
    if numel(w_full) ~= size(Xn,2)
        error('w_full 的长度 (%d) 与标准化后特征列数 (%d) 不一致。请替换正确的 w_full。', numel(w_full), size(Xn,2));
    end
    L = Xn * w_full(:) + bias;
end

%% --------- 5) 输出 ----------
yhat = G + L;

% 打印示例输入和输出（便于校验）
fprintf('predict_explicit_17: 返回 %d 个预测值。示例 yhat(1) = %.6g\n', numel(yhat), yhat(1));
fprintf('示例完整输入 Xfull(1,:) = [ '); fprintf('%.6g ', Xfull(1,:)); fprintf(']\n');

end


