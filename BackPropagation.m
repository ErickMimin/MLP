function newParam = BackPropagation(mlpParam, alpha, a, target, vectorFunction)
%Derivadas de funciones para MLP
F{1}=@(x) 1;
F{2}=@(x) x * (1 - x);
F{3}=@(x) 1 - (x ^ 2);
%Asignacion
newParam{1} = mlpParam{1};
newParam{2} = mlpParam{2};
M = size(vectorFunction, 2);
s = cell(1, M);
%Calculamos las sensitividades
s{M} = (-2) * F{vectorFunction(M)}(diag(transpose(a{M + 1}))) * (target - a{M + 1});
newParam{1}{M} = newParam{1}{M} - alpha * s{M} * transpose(a{M});  
newParam{2}{M} = newParam{2}{M} - alpha * s{M};
for m = M-1 : -1 : 1
    s{m} =  F{vectorFunction(m)}(diag(transpose(a{m + 1}))) * transpose(mlpParam{1}{m + 1}) * s{m + 1};
    newParam{1}{m} = newParam{1}{m} - alpha * s{m} * transpose(a{m});  
    newParam{2}{m} = newParam{2}{m} - alpha * s{m}; 
end
end

