function T = readPerformance(filename, sheetName)
    T = readtable(filename, 'Sheet', sheetName); % 读入时自动取第一行做列名
    T = T(:, {'温度','压力','流速','输出'});      % 按列名取
end