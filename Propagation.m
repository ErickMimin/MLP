function a = Propagation(vectorFunction, mlpParam, p)
% Funciones para MLP
f{1}=@(x) purelin(x);
f{2}=@(x) logsig(x);
f{3}=@(x) tansig(x);
% Incializacion de a
a = cell(size(vectorFunction, 2) + 1, 1);
% Valor de a para m = 0
a{1} = p;
for m = 1 : size(vectorFunction, 2)
    a{m + 1} = f{vectorFunction(1, m)}(mlpParam{1}{m}*a{m}+mlpParam{2}{m});
end
end

