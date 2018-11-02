function [S,L_hp,Lhp,Lhpr,Lhpg,Lhpb] = L1QP_FeatureSign_Set_wang(X, B, hDim, Sigma, beta, gamma, lammda2)

%hypergraph
K = 20;
Datatype = 'Line_expand';
GraphConstructMethod = 'AdaptiveKNN_Global';
L_hp  =  0;
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
    %一维求hyperspectral
    Y        = Xh(:,jj);
    Y        = Y';
    row      = 75;
    col      = 1;
    L        = HyperGraphCal_wu(Y,row,col,K,Datatype,GraphConstructMethod);
    %三维求hyperspectral 
    
%     Y        = Xh(:,jj);
%     Y=reshape(Y,25,3);
%     Y=Y';  
%     row      = 25;
%     col      = 1;
%     
%     L        = HyperGraphCal_wu(Y,row,col,K,Datatype,GraphConstructMethod);
     
    
    A        = Dh' * Dh + Dl' * Dl +  lammda2 * 2  * Dh' * L * Dh  + 2*beta*Sigma;
    
    b        = - (Dh'  *  Xh(  :  , jj)  +  Dl'  *  Xl(  :  , jj) );
    
    S(:, jj) = L1QP_FeatureSign_yang(gamma, A, b);% min  0.5*x'*A*x+b'*x+\lambda*|x|
    
    L_hp     = L_hp + lammda2 * (Dh * S(:, jj))' * L * (Dh * S(:, jj));
    
    Lhp{jj}  = L;
    Lhpr{jj}  = L(       1:hDim/3     , 1:hDim/3);
    Lhpg{jj}  = L(hDim/3+1 : 2*hDim/3 , hDim/3+1 : 2*hDim/3);
    Lhpb{jj}  = L(2*hDim/3+1 : hDim   , 2*hDim/3+1 : hDim);
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





