%Limpiamos datos
clear;clc;close all;mkdir('.', 'param')
%====================== Introduccion de datos ======================%
%Archivo de entrada 
inputFile = '01_Polinomio_Entrada';%input('Ingrese el nombre del archivo con los datos de entrada P(.txt): ', 's');
p = load(strcat(inputFile, '.txt'));
%Archivo de targets
targetFile = '01_Polinomio_Target';%input('Ingrese el nombre del archivo con los targets(.txt): ', 's');
target = load(strcat(targetFile, '.txt'));
% %Rango de la señal
% range = input('Ingrese el rango de la senal a aproximar [rango-minimo , rango-maximo]:');
%Arquitectura del MLP
layerVector = [1 8 1];%input('Ingrese el vector de entradas de cada capa [1,S^1,S^2,...,S^n,1]: ');
functionVector = [3 1];%input('Ingrese el vector de funciones de cada capa [1,2,3,...,2,1,3]: ');
%Factor de aprendizaje
learningRate = 1 * 10^(-2);%input('Ingrese el factor de aprendizaje dentro del rango 0 a 1: ');
%Condiciones de finalizacion
epochmax = input('Ingrese el numero maximo de epocas (epochmax): ');
min_error_train = input('Ingrese el valor minimo de error aceptado: ');
%Configuracion de validacion
epochval = input('Ingrese el numero de epocas de entrenamiento entre cada epoca de validacion: ');
numval = input('Ingrese el numero maximo de incrementos consecutivos de error de validacion: ');
aleat = input('¿Desea inicializar los valores de pesos y bias aleatoriamente?(1=Si/0=No)');
% Una prueba de backpropagation
% erick = BackPropagation({{[-0.27;-0.41], [0.09 -0.17]},{[-0.48;-0.13], 0.48}}, 0.1, {1;[0.321;0.368];0.446}, 1.261 + 0.446,[2 1]);
% erick2 = Propagation([2 1], {{[-0.27;-0.41], [0.09 -0.17]},{[-0.48;-0.13], 0.48}}, 1);
%====================== Inicio del programa ======================%
%Celdas para pesos y bias
mlpParam = cell(1 ,2);
%1 .- Celda de pesos
mlpParam{1} = cell(size(layerVector, 2) - 1, 1);
%2 .- Celda de bias
mlpParam{2} = cell(size(layerVector, 2) - 1, 1);
if aleat == 1
    %Generacion aleatoria de pesos y bias
    for i = 2 : size(layerVector, 2)
        mlpParam{1}{i - 1} = -1 + (1+1) * rand(layerVector(i), layerVector(i - 1));
        mlpParam{2}{i - 1} = -1 + (1+1) * rand(layerVector(i), 1);
    end
else
    for i = 2:size(layerVector, 2)
        %Cargar pesos
        j = i - 1;
        fileID = fopen("./param/PesosFinal" + (i-1) + ".txt", 'r');
        mlpParam{1, 1}{i-1, 1} = fscanf(fileID, '%f', [layerVector(i) Inf]);
        fclose(fileID);
        
        %Cargar bias
        fileID = fopen("./param/BiasFinal" + (i-1) + ".txt", 'r');
        mlpParam{1, 2}{i-1, 1} = fscanf(fileID, '%f', [layerVector(i) Inf]);
        fclose(fileID);
    end
end

for i = 1:size(layerVector, 2)-1
    %Guardar pesos
    fileID = fopen("./param/Pesos" + i + ".txt", 'w');
    fprintf(fileID, '%d ', mlpParam{1, 1}{i, 1});
    fclose(fileID);
    %Guardar bias
    fileID = fopen("./param/Bias" + i + ".txt", 'w');
    fprintf(fileID, '%d ', mlpParam{1, 2}{i, 1});
    fclose(fileID);
end


%Ahora vamos a separar los datos, usamos la funcion separarDatos pasando
%como parametros el arreglo/matriz de inputs y el arreglo/matriz de files
%nos regresa 3 matrices 2 filan x n columnas (depende del numero de datos
%en total
%para acceder a los datos Ejemplo conjValidacion(1,x) dato input y
%conjvaldacion(2,x) target respectivo


[conjAprendizaje,conjValidacion,conjPrueba] = SeparateData(p,target);

a = cell(size(conjAprendizaje, 2), 1);
errorEpoch = zeros(1);
for epoch = 1: epochmax
    errorEpoch(epoch) = 0;
    if(mod(epoch, epochval) == 0)
       % Epoca de validacion
       for i = 1:size(conjValidacion, 2)
           val = Propagation(functionVector, mlpParam(end, :), conjValidacion(1, i));
           % Error de epoca
           errorEpoch(epoch) = errorEpoch(epoch) + abs(conjValidacion(2, i) - val{end});
       end
       errorEpoch(epoch) = errorEpoch(epoch) / size(conjValidacion, 2);
       % Early stopping
       if(epoch - epochval > 0 && errorEpoch(epoch) > errorEpoch(epoch - epochval))
           increases = increases + 1;
       else
           increases = 0;
           mlpParamBeIn = mlpParam(end, :);
       end
       %increases = EarlyStopping(functionVector, mlpParam(end, :), conjValidacion);
       if(increases == numval)
           mlpParam(1, :) = mlpParamBeIn;
           fprintf("EarlyStopping paro el programa\n" + epoch);
           break;
       end
    else
        for i = 1:size(conjAprendizaje, 2)
           a{i} = Propagation(functionVector, mlpParam(end, :), conjAprendizaje(1, i));
           if(~isequal(conjAprendizaje(2, i), a{i}{end}))
               mlpParam(1, :) = BackPropagation(mlpParam(end, :), learningRate, a{i}, conjAprendizaje(2, i), functionVector);
           end
           % Suma de errores
           errorEpoch(epoch) = errorEpoch(epoch) + abs(conjAprendizaje(2, i) - a{i}{end});
        end
        % Error de la epoca
        errorEpoch(epoch) = errorEpoch(epoch) / size(conjAprendizaje, 2);
        % Validacion de error minimo
        if(errorEpoch(epoch) < min_error_train)
            fprintf("Minimo error de entrenamiento aceptado\n");
            break;
        end
    end
    if(mod(epoch, 100) == 0)
        disp("Epoca: " + epoch);
        for i = 1:size(layerVector, 2)-1
            %Guardar pesos
            fileID = fopen("./param/Pesos" + i + ".txt", 'a');
            fprintf(fileID, '%d ', mlpParam{1, 1}{i, 1});
            fclose(fileID);
            %Guardar bias
            fileID = fopen("./param/Bias" + i + ".txt", 'a');
            fprintf(fileID, '%d ', mlpParam{1, 2}{i, 1});
            fclose(fileID);
        end
    end
end

%Guardar pesos y bias finales
for i = 1:size(layerVector, 2)-1
    %Guardar pesos
    fileID = fopen("./param/PesosFinal" + i + ".txt", 'w');
    fprintf(fileID, '%d ', mlpParam{1, 1}{i, 1});
    fclose(fileID);
    %Guardar bias
    fileID = fopen("./param/BiasFinal" + i + ".txt", 'w');
    fprintf(fileID, '%d ', mlpParam{1, 2}{i, 1});
    fclose(fileID);
end

% Epoca de prueba
aTest = cell(size(conjPrueba, 2), 1);
errorTest = 0;
for i = 1:size(conjPrueba, 2)
   aTest{i} = Propagation(functionVector, mlpParam(end, :), conjPrueba(1, i));
   errorTest = errorTest + abs(conjPrueba(2, i) - aTest{i}{end});
end
errorTest = errorTest / size(conjPrueba, 2);

% Grafica errores de entranamiento y validacion
errorEpochT = zeros(2, 1);
carry = 0;
for i = 1:length(errorEpoch)
    if(mod(i, epochval) ~= 0)
        carry = carry + 1;
        errorEpochT(2, carry) = errorEpoch(1, i);
        errorEpochT(1, carry) = i;
    end
end
figure
hold on;
plot(errorEpoch, 'or','MarkerIndices',1:epochval:length(errorEpoch));
plot(errorEpochT(1,:),errorEpochT(2,:));
title("Errores de entranmiento y validación");
xlabel('Época');
ylabel('Error');
hold off;

% Grafica del conjunto de pruebas
aTestG = zeros(1);
for i = 1:size(conjPrueba, 2)
   aTestG(1, i) = aTest{i}{end};
end
figure
hold on;
plot(conjPrueba(1, :), aTestG(1, :), 'x');
plot(conjPrueba(1, :), conjPrueba(2, :), 'o');
title("Target vs Salida");
xlabel('X');
ylabel('Y');
hold off;
figure
hold on;
plot(conjPrueba(1, :), aTestG(1, :));
plot(conjPrueba(1, :), conjPrueba(2, :));
hold off;



