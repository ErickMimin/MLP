%Limpiamos datos
clear;clc;close all;mkdir('.', 'param')
%====================== Introduccion de datos ======================%
%Archivo de entrada 
inputFile = input('Ingrese el nombre del archivo con los datos de entrada P(.txt): ', 's');
p = load(strcat(inputFile, '.txt'));
%Archivo de targets
targetFile =  input('Ingrese el nombre del archivo con los targets(.txt): ', 's');
target = load(strcat(targetFile, '.txt'));
% %Rango de la señal
% range = input('Ingrese el rango de la senal a aproximar [rango-minimo , rango-maximo]:');
%Arquitectura del MLP
layerVector = input('Ingrese el vector de entradas de cada capa [1,S^1,S^2,...,S^n,1]: ');
functionVector = input('Ingrese el vector de funciones de cada capa [1,2,3,...,2,1,3]: ');
%Factor de aprendizaje
learningRate = input('Ingrese el factor de aprendizaje dentro del rango 0 a 1: ');
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
% Incializacion para errores
errorEpochT = zeros(1);
errorEpochV = zeros(1);
carryT = 0;
carryV = 0;
% Inicio de epocas
for epoch = 1: epochmax
    if(mod(epoch, epochval) == 0)
       % Epoca de validacion
       carryV = carryV + 1;
       errorEpochV(2, carryV) = 0;
       for i = 1:size(conjValidacion, 2)
           val = Propagation(functionVector, mlpParam(end, :), conjValidacion(1, i));
           % Error de epoca
           errorEpochV(2, carryV) = errorEpochV(2, carryV) + abs(conjValidacion(2, i) - val{end});
       end
       % Error de validacion
       errorEpochV(2, carryV) = errorEpochV(2, carryV) / size(conjValidacion, 2);
       errorEpochV(1, carryV) = epoch;
       % Early stopping
       if(carryV > 1 && errorEpochV(2, carryV) > errorEpochV(2, carryV - 1))
           increases = increases + 1;
       else
           increases = 0;
           mlpParamBeIn = mlpParam(end, :);
       end
       %increases = EarlyStopping(functionVector, mlpParam(end, :), conjValidacion);
       if(increases == numval)
           mlpParam(1, :) = mlpParamBeIn;
           fprintf("EarlyStopping paro el programa\n");
           break;
       end
    else
        carryT = carryT + 1;
        errorEpochT(2, carryT) = 0;
        for i = 1:size(conjAprendizaje, 2)
           a{i} = Propagation(functionVector, mlpParam(end, :), conjAprendizaje(1, i));
           if(~isequal(conjAprendizaje(2, i), a{i}{end}))
               mlpParam(1, :) = BackPropagation(mlpParam(end, :), learningRate, a{i}, conjAprendizaje(2, i), functionVector);
           end
           % Suma de errores
           errorEpochT(2, carryT) = errorEpochT(2, carryT) + abs(conjAprendizaje(2, i) - a{i}{end});
        end
        % Error de la epoca
        errorEpochT(2, carryT) = errorEpochT(2, carryT) / size(conjAprendizaje, 2);
        errorEpochT(1, carryT) = epoch;
        % Validacion de error minimo
        if(errorEpochT(2, carryT) < min_error_train)
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
figure
hold on;
plot(errorEpochV(1, :), errorEpochV(2, :), 'or');
plot(errorEpochT(1, :), errorEpochT(2, :));
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


% Grafica de la funcion
aGraph = cell(1);
aG = zeros(size(p, 1), 1);
for i = 1:size(p, 1)
   aGraph{i} = Propagation(functionVector, mlpParam(end, :), p(i, 1));
   aG(i, 1) = aGraph{i}{end};
end

figure
hold on;
plot(p(:, 1), target(:, 1), '-b');
plot(p(:, 1), aG(:, 1), '-r');
title("Function");
xlabel('X');
ylabel('Y');
hold off;

W = cell([1 size(layerVector, 2)-1]);
B = cell([1 size(layerVector, 2)-1]);

n = 0;
for i = 1:size(layerVector, 2)-1
    datos = load("./param/Pesos"+i+".txt");
    Waux = vec2mat(datos,layerVector(i)*layerVector(i+1));
    n = n + layerVector(i)*layerVector(i+1);
    W{i} = Waux;
    datos = load("./param/Bias"+i+".txt");
    Baux = vec2mat(datos,layerVector(i+1));
    n = n + layerVector(i+1);
    B{i} = Baux;
    
    disp("W"+i);
    disp(W{i});
    disp("B"+i);
    disp(B{i});
end

figure
hold on;

disp("n");
disp(n);
leyendas = strings([1,n]);

for i = 1:size(W, 2)
    for j = 1:size(W{i},2)
        plot(1:size(W{i}(:,j),1),W{i}(:,j).','-o');
        leyendas((i-1)*size(W{i},2)+j) = sprintf('W(%d,%d)',i,j);
        n = (i-1)*size(W{i},2)+j;
    end
end

for i = 1:size(B, 2)
    for j = 1:size(B{i},2)
        plot(1:size(B{i}(:,j),1),B{i}(:,j).','-o');
        leyendas((i-1)*size(W{i},2)+j+n) = sprintf('B(%d,%d)',i,j);
    end
end
disp("leyendas");
disp(leyendas);

title("Evolution");
legend(leyendas,'Location','northeastoutside');
xlabel('epocas');
ylabel('valor');
hold off;

fprintf("Error Entrenamiento: %f\n", errorEpochT(2, end));
fprintf("Error Validacion: %f\n", errorEpochV(2, end));
fprintf("Error Entrenamiento: %f\n", errorTest);