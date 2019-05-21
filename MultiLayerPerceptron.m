%Limpiamos datos
clear;clc;close all;
%Archivo de entrada 
inputFile = 'Dataset';%input('Ingrese el nombre del archivo con los datos de entrada P(.txt): ', 's');
p = load(strcat(inputFile, '.txt'));
%Archivo de targets
targetFile = 'Target';%input('Ingrese el nombre del archivo con los targets(.txt): ', 's');
target = load(strcat(targetFile, '.txt'));
% %Rango de la señal
% range = input('Ingrese el rango de la senal a aproximar [rango-minimo , rango-maximo]:');
%Arquitectura del MLP
layerVector = [1 2 1];%input('Ingrese el vector de entradas de cada capa [1,S^1,S^2,...,S^n,1]: ');
functionVector = [1 2];%input('Ingrese el vector de funciones de cada capa [1,2,3,...,2,1,3]: ');
%Factor de aprendizaje
learningRate = 0.1;%input('Ingrese el factor de aprendizaje dentro del rango 0 a 1: ');
%Condiciones de finalizacion
epochmax = input('Ingrese el numero maximo de epocas (epochmax): ');
%min_error_train = input('Ingrese el valor maximo de error aceptado: ');
%Configuracion de validacion
epochval = input('Ingrese el numero de epocas de entrenamiento entre cada epoca de validacion: ');
numval = input('Ingrese el numero maximo de incrementos consecutivos de error de validacion: ');
%Celdas para pesos y bias
mlpParam = cell(1 ,2);
%1 .- Celda de pesos
mlpParam{1} = cell(size(layerVector, 2) - 1, 1);
%2 .- Celda de bias
mlpParam{2} = cell(size(layerVector, 2) - 1, 1);
%Generacion aleatoria de pesos y bias
for i = 2 : size(layerVector, 2)
    mlpParam{1}{i - 1} = -1 + (1+1) * rand(layerVector(i), layerVector(i - 1));
    mlpParam{2}{i - 1} = -1 + (1+1) * rand(layerVector(i), 1);
end
%Ahora vamos a separar los datos, usamos la funcion separarDatos pasando
%como parametros el arreglo/matriz de inputs y el arreglo/matriz de files
%nos regresa 3 matrices 2 filan x n columnas (depende del numero de datos
%en total
%para acceder a los datos Ejemplo conjValidacion(1,x) dato input y
%conjvaldacion(2,x) target respectivo
[conjAprendizaje,conjValidacion,conjPrueba] = separarDatos(p,target);
for epoch = 1: epochmax
    if(mod(epoch, epochval) == 0)
       increases = EarlyStopping(functionVector, mlpParam(end, :), conjValidacion);
       if(increases == numval)
           fprintf("EarlyStopping paro el programa");
           break;
       end
    else
        a = cell(size(conjAprendizaje, 2), 1);
        for i = 1:size(conjAprendizaje, 2)
           a{i} = Propagation(functionVector, mlpParam(end, :), conjAprendizaje(1, i)); 
           if(~isequal(conjAprendizaje(2, i), a{i}{end}))
               mlpParam(size(mlpParam, 1) + 1, :) = BackPropagation(mlpParam(end, :), learningRate, a{i}, conjAprendizaje(2, i), functionVector);
           end
        end
    end
end




