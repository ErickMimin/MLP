function increases = EarlyStopping(functionVector, mlpParam, validation,target)
    for i=1:size(validation, 1)
        a = Propagation(functionVector, mlpParam(end, :), transpose(validation(i, :)));
        if(i==1)
            error = abs(target(i)-a);
        end
        
        if(abs(target(i)-a)>error)
            increases = increases + 1;
        else
            increases = 0;
        end
        
        error = abs(target(i)-a);
    end
end

