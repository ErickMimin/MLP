function [aprendizaje,validacion,prueba] = separarDatos(inputs, targets)
[sz,m] = size(inputs);
if sz>100
    sz_a = round(sz * .8);
    sz_v = round(sz * .1);
    sz_p = round(sz * .1);
    proporcion = round(((sz - 2) * .8) / ((sz - 2) * .2 + 1));
else
    sz_a = round(sz * .7);
    sz_v = round(sz * .15);
    sz_p = round(sz * .15);
    proporcion = floor(((sz - 2) * .7) / ((sz - 2) * .3 + 1));
end

t = 1+sz_v*(2*proporcion+2)+proporcion+1;
e = t-sz;

aprendizaje = zeros(2,sz_a);
ia = 2;
validacion = zeros(2,sz_v);
iv = 1;
prueba = zeros(2,sz_p);
ip = 1;

aprendizaje(1,1) = inputs(1);
aprendizaje(2,1) = targets(1);
for i = 2:sz-1
    n = proporcion;
    
    if e>0
        n = n - 1;
    end
    if mod(i,2*(n+1)) == 1 && ip<=sz_p
        prueba(1,ip) = inputs(i);
        prueba(2,ip) = targets(i);
        ip = ip+1;
        if n == 3
            e = e-1;
        end
    elseif mod(i,2*(n+1)) == 6 && iv<=sz_v
        validacion(1,iv) = inputs(i);
        validacion(2,iv) = targets(i);
        iv = iv+1;
    else
        aprendizaje(1, ia) = inputs(i);
        aprendizaje(2,ia) = targets(i);
        ia = ia+1;
    end
end
aprendizaje(1,ia) = inputs(sz);
aprendizaje(2,ia) = targets(sz);
end

