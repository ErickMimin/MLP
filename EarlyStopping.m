function increases = EarlyStopping(functionVector, mlpParam, validation)
    for i=1:size(validation, 2)
        a = Propagation(functionVector, mlpParam, transpose(validation(1, i)));
        if(i==1)
            error = abs(validation(2, i)-a{end});% t - a
        end
        
        if(abs(validation(2, i)- a{end}) > error)
            increases = increases + 1;
        else
            increases = 0;
        end
        
        error = abs(validation(2, i)-a{end});
    end
end

