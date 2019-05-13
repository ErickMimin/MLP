function a = Propagation(vectorFunction, W, b, p)
f{1}=@(x) purelin(x);
f{2}=@(x) logsig(x);
f{3}=@(x) tansig(x);
a = p;
for m = 1 : size(vectorFunction, 2)
    a = f{vectorFunction(1, m)}(W{m}*a+b{m});
end

