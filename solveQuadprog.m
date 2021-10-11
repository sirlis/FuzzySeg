function mat_f = solveQuadprog(s_mask,mat_normLap)

s_length = size(s_mask,1);
lb = zeros(s_length, 1) - 1 + 2*double(s_mask==1);
ub = zeros(s_length, 1) + 1 - 2*double(s_mask==-1);
options = optimset('Algorithm','interior-point-convex');
options = optimset(options,'Display','iter','TolFun',1e-8);
mat_f = quadprog(mat_normLap,[],[],[],[],[],lb,ub,[],options);
mat_f(mat_f<=0) = -1;
mat_f(mat_f>0) = 1;
    
end