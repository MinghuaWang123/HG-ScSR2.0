function [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc(X, B,  S, hDim, Sigma, beta, gamma,L_hp)

nBases = size(B, 2);
% Err = X - B*S;
Errh = X(  1   : hDim , :) - B(    1   : hDim , :) * S;
Errl = X( hDim+1 : end  , :) - B( hDim+1 : end  , :) * S;

fresidue = 0.5*sum(sum(Errh.^2))+0.5*sum(sum(Errl.^2))+L_hp;

fsparsity = gamma*sum(sum(abs(S)));

fregs = 0;
for ii = size(S, 2),%??origin size(S,1)
    fregs = fregs + beta*S(:, ii)'*Sigma*S(:, ii);
end

fobj = fresidue + fsparsity + fregs;