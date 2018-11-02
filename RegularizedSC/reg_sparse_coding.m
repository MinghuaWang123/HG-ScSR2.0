function [B, S, stat] = reg_sparse_coding(X, hDim,lDim,num_bases, Sigma, beta, gamma, lammda2, prou, num_iters, batch_size, initB, fname_save)
%
% Regularized sparse coding
%
% Inputs
%       X           -data samples, column wise
%       num_bases   -number of bases
%       Sigma       -smoothing matrix for regularization
%       beta        -smoothing regularization
%       gamma       -sparsity regularization
%       num_iters   -number of iterations 
%       batch_size  -batch size
%       initB       -initial dictionary
%       fname_save  -file name to save dictionary
%
% Outputs
%       B           -learned dictionary
%       S           -sparse codes
%       stat        -statistics about the training
%
% Written by Jianchao Yang @ IFP UIUC, Sep. 2009.

pars = struct;
pars.patch_size = size(X,1);
pars.num_patches = size(X,2);
pars.num_bases = num_bases;
pars.num_trials = num_iters;
pars.beta = beta;
pars.gamma = gamma;
pars.VAR_basis = 1; % maximum L2 norm of each dictionary atom

if ~isa(X, 'double'),
    X = cast(X, 'double');
end

if isempty(Sigma),
	Sigma = eye(pars.num_bases*3);
end

if exist('batch_size', 'var') && ~isempty(batch_size)
    pars.batch_size = batch_size; 
else
    pars.batch_size = size(X, 2);
end

if exist('fname_save', 'var') && ~isempty(fname_save)
    pars.filename = fname_save;
else
    pars.filename = sprintf('Results/reg_sc_b%d_%s', num_bases, datestr(now, 30));	
end

pars

% initialize basis
if ~exist('initB') || isempty(initB)
    Bh1 = rand(hDim/3, pars.num_bases)-0.5;
	Bh1 = Bh1 - repmat(mean(Bh1,1), size(Bh1,1),1);
    Bh1 = Bh1*diag(1./sqrt(sum(Bh1.*Bh1)));
    
    Bh2 = rand(hDim/3, pars.num_bases)-0.5;
	Bh2 = Bh2 - repmat(mean(Bh2,1), size(Bh2,1),1);
    Bh2 = Bh2*diag(1./sqrt(sum(Bh2.*Bh2)));

    Bh3 = rand(hDim/3, pars.num_bases)-0.5;
	Bh3 = Bh3 - repmat(mean(Bh3,1), size(Bh3,1),1);
    Bh3 = Bh3*diag(1./sqrt(sum(Bh3.*Bh3)));
    
    Bl1 = rand(lDim/3, pars.num_bases)-0.5;
	Bl1 = Bl1 - repmat(mean(Bl1,1), size(Bl1,1),1);
    Bl1 = Bl1*diag(1./sqrt(sum(Bl1.*Bl1)));
    
    Bl2 = rand(lDim/3, pars.num_bases)-0.5;
	Bl2 = Bl2 - repmat(mean(Bl2,1), size(Bl2,1),1);
    Bl2 = Bl2*diag(1./sqrt(sum(Bl2.*Bl2)));
    
    Bl3 = rand(lDim/3, pars.num_bases)-0.5;
	Bl3 = Bl3 - repmat(mean(Bl3,1), size(Bl3,1),1);
    Bl3 = Bl3*diag(1./sqrt(sum(Bl3.*Bl3)));
    
    Blr  = Bl1;
    Blg  = Bl2;
    Blb  = Bl3;
    
else
    disp('Using initial B...');
    B = initB;
end
% 
% Bh2=Bh1;Bh3=Bh1;
% 
% Bl2=Bl1;Bl3=Bl1;

B=zeros(pars.patch_size, pars.num_bases*3);
B(        1        :  hDim/3         ,  1                  :  pars.num_bases)       =Bh1;
B(     hDim/3+1    :  2*hDim/3       ,  pars.num_bases+1   :  pars.num_bases*2)     =Bh1;
B(    2*hDim/3+1   :  hDim           ,  2*pars.num_bases+1 :  pars.num_bases*3)     =Bh1;

B(      hDim+1     :  hDim+lDim/3    ,  1                  :  pars.num_bases)       =Bl1;
B(  hDim+lDim/3+1  :  hDim+2*lDim/3  ,  pars.num_bases+1   :  pars.num_bases*2)     =Bl1;
B( hDim+2*lDim/3+1 :  hDim+lDim      ,  2*pars.num_bases+1 :  pars.num_bases*3)     =Bl1;

[L M]=size(B);%B应为字典初始化

t=0;
% statistics variable
stat= [];
stat.fobj_avg = [];
stat.elapsed_time=0;

		opts.max_iter      = 500;
		opts.show_progress = 0;
		opts.check_grad    = false;  
		opts.tol           = 1e-8;  
		opts.verbose     = true;
        optsD = opts;
	    optsD.max_iter = 200;
	    optsD.tol      = 1e-8;
        
% optimization loop
while t < pars.num_trials
    t=t+1;
    start_time= cputime;
    stat.fobj_total=0;    
    % Take a random permutation置换 of the samples
    indperm = randperm(size(X,2));
    
    sparsity = [];
    
    for batch=1:(size(X,2)/pars.batch_size),
        % This is data to use for this step
        batch_idx = indperm((1:pars.batch_size)+pars.batch_size*(batch-1));
        Xb = X(:,batch_idx);%又将X按列乱序了
        
        % learn coefficients (conjugate gradient)稀疏系数
        %[S, L_hp, Lhp] = L1QP_FeatureSign_Set_wang(Xb, B, hDim, Sigma, pars.beta, pars.gamma, lammda2);
        [S, L_hp,Lhp,Lhpr,Lhpg,Lhpb] = L1QP_FeatureSign_Set_wang(Xb, B, hDim, Sigma, pars.beta, pars.gamma, lammda2);
        
        sparsity(end+1) = length(find(S(:) ~= 0))/length(S(:));%稀疏系数的稀疏度
        
        % get objective
        [fobj] = getObjective_RegSc(Xb, B, S, hDim, Sigma, pars.beta, pars.gamma, L_hp);       
        stat.fobj_total = stat.fobj_total + fobj;%更新完稀疏系数之后的代价函数的值 0.5(（xDc*Z）^2)^0.5
        % update basis
%         [Bhr,Bhg,Bhb] = l2ls_learn_basis_dual_wADMM(B(1:hDim , 1:3*pars.num_bases), Xb, S, lammda2, pars.num_bases,  hDim, Lhp,Lhpr,Lhpg,Lhpb, prou);%pars.VAR_basis=1;求解字典B=Dc
        [Bh] = l2ls_learn_basis_dual_wmhADMM(B(1:hDim ,:), Xb(1:hDim ,:), S, lammda2, pars.num_bases,  hDim, Lhp, prou);%pars.VAR_basis=1;求解字典B=Dc

        %Bhr = l2ls_learn_basis_dual(Xb(       1         :    hDim/3      ,  :  ), S(        1          : pars.num_bases    ,  :), pars.VAR_basis);%pars.VAR_basis=1;求解字典B=Dc
        %Bhg = l2ls_learn_basis_dual(Xb(   hDim/3+1      :    2*hDim/3    ,  :  ), S(  pars.num_bases+1 : 2*pars.num_bases  ,  :), pars.VAR_basis);
        %Bhb = l2ls_learn_basis_dual(Xb(   2*hDim/3+1    :    hDim        ,  :  ), S(2*pars.num_bases+1 : 3*pars.num_bases  ,  :), pars.VAR_basis);

        Blr  = l2ls_learn_basis_dual(Xb(     hDim+1      :  hDim+lDim/3   ,  :  ), S(        1          : pars.num_bases    ,  :), pars.VAR_basis);
        Blg  = l2ls_learn_basis_dual(Xb(  hDim+lDim/3+1  :  hDim+2*lDim/3 ,  :  ), S(  pars.num_bases+1 : 2*pars.num_bases  ,  :), pars.VAR_basis);
        Blb  = l2ls_learn_basis_dual(Xb( hDim+2*lDim/3+1 :  hDim+lDim     ,  :  ), S(2*pars.num_bases+1 : 3*pars.num_bases  ,  :), pars.VAR_basis);
       
%         Xr= Xb(     hDim+1      :  hDim+lDim/3   ,  :  );  
%         Sr= S(        1          : pars.num_bases    ,  :);
%         Fr = Sr*Sr'; Er = Xr*Sr';
%         Blr  = ODL_updateD(Blr,Fr,Er,optsD);
%         
%         Xg= Xb(  hDim+lDim/3+1  :  hDim+2*lDim/3 ,  :  );  
%         Sg= Xb(  pars.num_bases+1 : 2*pars.num_bases  ,  :);
%         Fg = Sg*Sg'; Eg = Xg*Sg';
%         Blg  = ODL_updateD(Blg,Fg,Eg,optsD);
%         
%         Xb= Xb( hDim+2*lDim/3+1 :  hDim+lDim     ,  :  );  
%         Sb=S(2*pars.num_bases+1 : 3*pars.num_bases  ,  :);
%         Fb = Sb*Sb'; Eb = Xb*Sb';
%         Blb  = ODL_updateD(Blb,Fb,Eb,optsD);
        
%         B(        1        :  hDim/3         ,           1         :  pars.num_bases)       =Bhr;
%         B(   hDim/3+1      :  2*hDim/3       ,  pars.num_bases+1   :  2*pars.num_bases)     =Bhg;
%         B(  2*hDim/3+1     :  hDim           ,  2*pars.num_bases+1 :  3*pars.num_bases)     =Bhb;

        B(        1        :  hDim/3         ,           1         :  pars.num_bases)       =Bh(        1        :  hDim/3         ,           1         :  pars.num_bases);
        B(   hDim/3+1      :  2*hDim/3       ,  pars.num_bases+1   :  2*pars.num_bases)     =Bh(   hDim/3+1      :  2*hDim/3       ,  pars.num_bases+1   :  2*pars.num_bases);
        B(  2*hDim/3+1     :  hDim           ,  2*pars.num_bases+1 :  3*pars.num_bases)     =Bh(  2*hDim/3+1     :  hDim           ,  2*pars.num_bases+1 :  3*pars.num_bases);       
       
        B(      hDim+1     :  hDim+lDim/3    ,           1         :  pars.num_bases)       =Blr;
        B(  hDim+lDim/3+1  :  hDim+2*lDim/3  ,  pars.num_bases+1   :  2*pars.num_bases)     =Blg;
        B( hDim+2*lDim/3+1 :  hDim+lDim      ,  2*pars.num_bases+1 :  3*pars.num_bases)     =Blb;
  
    end
    
    % get statistics
    stat.fobj_avg(t)      = stat.fobj_total / pars.num_patches;
    stat.elapsed_time(t)  = cputime - start_time;
    fobj1(t)              = fobj/pars.num_patches;
    sp(t)                 = mean(sparsity);
    fprintf(['epoch= %d, sparsity = %f, fobj= %f, fobj1= %f, took %0.2f ' ...
             'seconds\n'], t, mean(sparsity), stat.fobj_avg(t), fobj1(t), stat.elapsed_time(t));
         
    % save results
    fprintf('saving results ...\n');
    experiment = [];
    experiment.matfname = sprintf('%s.mat', pars.filename);     
    save(experiment.matfname, 't', 'pars', 'B', 'stat');
    fprintf('saved as %s\n', experiment.matfname);
end
%      Bhr=Bhr./repmat(sqrt(sum(Bhr.^2, 1)), hDim/3, 1);
%      Bhg=Bhg./repmat(sqrt(sum(Bhg.^2, 1)), hDim/3, 1);
%      Bhb=Bhb./repmat(sqrt(sum(Bhb.^2, 1)), hDim/3, 1);
%      
%      Blr=Blr./repmat(sqrt(sum(Blr.^2, 1)), lDim/3, 1);
%      Blg=Blg./repmat(sqrt(sum(Blg.^2, 1)), lDim/3, 1);
%      Blb=Blb./repmat(sqrt(sum(Blb.^2, 1)), lDim/3, 1);
save stat.fobj_avg.mat;
save fobj1;
return

function retval = assert(expr)
retval = true;
if ~expr 
    error('Assertion failed');
    retval = false;
end
return
