%Limpiamos datos
clear;clc;close all;
%Archivo de entrada 
inputFile = input('Ingrese el nombre del archivo con los datos de entrada P(.txt): ', 's');
p = load(strcat('Dataset.txt'));
%Archivo de targets
targetFile = input('Ingrese el nombre del archivo con los targets(.txt): ', 's');
%target = load(strcat(targetFile, '\.txt'));
% %Rango de la señal
% range = input('Ingrese el rango de la senal a aproximar [rango-minimo , rango-maximo]:');
%Arquitectura del MLP
layerVector = input('Ingrese el vector de entradas de cada capa [1,S^1,S^2,...,S^n,1]: ');
functionVector = input('Ingrese el vector de funciones de cada capa [1,2,3,...,2,1,3]: ');
%Factor de aprendizaje
% learningRate = input('Ingrese el factor de aprendizaje dentro del rango 0 a 1: ');
%Condiciones de finalizacion
% epochmax = input('Ingrese el numero maximo de epocas (epochmax): ');
% min_error_train = input('Ingrese el valor maximo de error aceptado: ');
%Configuracion de validacion
% epochval = input('Ingrese el numero de epocas de entrenamiento entre cada epoca de validacion: ');
%numval = input('Ingrese el numero maximo de incrementos consecutivos de error de validacion: ');
%Configuracion del dataset 
if(input('Ingrese 1) 80%-10%-10% o 2) 70%-15%-15%: ') == 1)
    trainingDataset = p(1:round(size(p, 1)*0.8), :);
    validationDataset = p(1:round(size(p, 1)*0.1), :);
    testDataset = p(1:round(size(p, 1)*0.1), :);
else
    trainingDataset = p(1:round(size(p, 1)*0.7), :);
    validationDataset = p(1:round(size(p, 1)*0.15), :);
    testDataset = p(1:round(size(p, 1)*0.15), :);
end
%Celdas para pesos y bias
W = cell(size(layerVector, 2) - 1, 1);
b = cell(size(layerVector, 2) - 1, 1);
%Generacion aleatoria de pesos y bias
for i = 2 : size(layerVector, 2)
    W{i - 1} = 1 + (1+1) * rand(layerVector(i), layerVector(i - 1));
    b{i - 1} = 1 + (1+1) * rand(layerVector(i), 1);
end
a = zeros(size(p, 1), 1);
%Propagamos todos los datos de P
for i = 1: size(p, 1)
   a(i, 1) = Propagation(functionVector, W, b, transpose(p(i)));      
end