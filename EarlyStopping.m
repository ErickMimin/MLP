function increases = EarlyStopping(functionVector, mlpParam, validation)
    for i=1:size(validation, 1)
        a = Propagation(functionVector, mlpParam(end, :), transpose(validation(i, :)));
        if(i==1)
            error = abs(validation(i, 2)-a);
        end
        
        if(abs(validation(i, 2)-a)>error)
            increases = increases + 1;
        else
            increases = 0;
        end
        
        error = abs(validation(i, 2)-a);
    end
end

