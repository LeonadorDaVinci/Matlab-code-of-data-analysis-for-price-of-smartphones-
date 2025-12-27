function T = readPrice(filename, sheetName)
    T = readtable(filename, 'Sheet', sheetName);
    T = T(:, {'Var1','Var2','Var3','Var4','Result'});
    T.Properties.VariableNames = {'A','B','C','D','Y'};
end