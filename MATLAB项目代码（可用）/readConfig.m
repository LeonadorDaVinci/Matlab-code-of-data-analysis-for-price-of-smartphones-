function T = readConfig(filename, sheetName)
    T = readtable(filename, 'Sheet', sheetName);
    T = T(:, {'X1','X2','Output'});
    T.Properties.VariableNames = {'A','B','Y'}; % 统一列名
end