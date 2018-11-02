function [S] = L1QP_FeatureSign_Set(X, B, hDim, Sigma, beta, gamma, lammda2)

[dFea, nSmp] = size(X);
nBases = size(B, 2);

% sparse codes of the features
S = sparse(nBases, nSmp);

%patch sparate
Xh = X(   1    :  hDim  ,       :      );
Xl = X( hDim+1 :  dFea  ,       :      );

%dictionary sparate
Dh = B(   1    :  hDim  ,       :      );
Dl = B( hDim+1 :  dFea  ,       :      );

%% compute the hypergraph
for jj = 1 : nSmp
    
    W        = constructW(Xh(  :  , jj));
    
    [m,n]    = size(Xh(  :  , jj));
    
    H        = sparse(m,m);
    
        for i=1:m
           for j=1:n
               if W(i,j) ~= 0
                 H(i,j)=1;  
               end
           end
        end
    
    invDe    = 1/6 * H;
    DCol     = full(sum(W,2));%Dv顶点的度
    Dv       = spdiags(DCol,0,speye(size(W,1)));%将Dcol变为对角线上元素，并且是稀疏矩阵
    L        = Dv - H*W*invDe*H';
    L1       = Dv - W;
     
    A        = Dh' * Dh + Dl' * Dl +  lammda2 * 2 * Dh' * L * Dh + 2*beta*Sigma;
    
    b        = - (Dh'  *  Xh(  :  , jj)  +  Dl'  *  Xl(  :  , jj) );
    
    S(:, jj) = L1QP_FeatureSign_yang(gamma, A, b);% min  0.5*x'*A*x+b'*x+\lambda*|x|
    

%     A1       = Dh' * Dh + Dl' * Dl + lammda2 * 2 * Dh' * L1 * Dh + 2*beta*Sigma;
%        
%     S1(:, jj)= L1QP_FeatureSign_yang(gamma, A1, b);
    
end   
    

% A = B'*B + 2*beta*Sigma;
% 
% for ii = 1:nSmp,
% %     b = -B'*X(:, ii);
% %     [net] = L1QP_FeatureSign(gamma, A, b);
% 
% 
% end




