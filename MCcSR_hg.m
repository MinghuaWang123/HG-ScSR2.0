% Enforcing edge Similarity.
function [hIm] = MCcSR_hg(lIm, up_scale, lambda1, lammda2,overlap, DicSize, patch_size, K)

% im_l_ycbcr = rgb2ycbcr(lIm);
 

%% load dictionary
% dict = ['Dictionary/hp_Color_D_' '_' num2str(nSmp) num2str(DicSize) '_' num2str(lammda2) '_' num2str(patch_size) '.mat' ];
% load(dict);

load('Dictionary/1yz20hp50_Color_D_20_100000_512_0.001_0.5_5_s3.mat');

Dho=Dh;
Dlo=Dl;

Dh_r = Dh(1        :   end/3    ,  1      :   end/3 );
Dh_g = Dh(  end/3+1: 2*end/3    ,  end/3+1: 2*end/3 );
Dh_b = Dh(2*end/3+1:   end      ,2*end/3+1:   end   );

Dl_r = Dl(1        :   end/3    ,  1      :   end/3 );
Dl_g = Dl(  end/3+1: 2*end/3    ,  end/3+1: 2*end/3 );
Dl_b = Dl(2*end/3+1:   end      ,2*end/3+1:   end   );

Dh = [Dh_r Dh_g Dh_b];
Dl = [Dl_r Dl_g Dl_b];

% normalize the dictionary
norm_Dl = sqrt(sum(Dl.^2, 1)); %自己平方，再列加和，开根号：1*1536
Dl = Dl./repmat(norm_Dl, size(Dl, 1), 1);%列归一化，repmat复制和平铺
patch_size = sqrt(size(Dh, 1));

%% bicubic interpolation of the low-resolution image
mIm = single(imresize(lIm, up_scale, 'bicubic'));
mIm_r = mIm(:,:,1);
mIm_g = mIm(:,:,2);
mIm_b = mIm(:,:,3);

im_m_ycbcr =  ( rgb2ycbcr(imresize(lIm, up_scale, 'bicubic')) );

h1 = [3 10 3 ; 0 0 0 ; -3 -10 -3 ]/16;
h2 = [3 0 -3 ; 10 0 -10 ; 3 0 -3 ]/16;

hIm = zeros(size(mIm));
hIm_r = zeros(size(mIm_r));
hIm_g = zeros(size(mIm_g));
hIm_b = zeros(size(mIm_b));

cntMat = zeros(size(mIm));
cntMat_r = zeros(size(mIm_r));
cntMat_g = zeros(size(mIm_g));
cntMat_b = zeros(size(mIm_b));

[h, w, dim] = size(mIm);

% extract low-resolution image features
lImfea_r = extr_lIm_fea(mIm(:,:,1));%做了一阶和二阶梯度滤波提取特征
lImfea_g = extr_lIm_fea(mIm(:,:,2));
lImfea_b = extr_lIm_fea(mIm(:,:,3));

% patch indexes for sparse recovery (avoid boundary)
gridx = 1:patch_size - overlap : w-patch_size;
gridx = 1:patch_size - overlap : w-patch_size+1;
gridy = 1:patch_size - overlap : h-patch_size;
gridy = 1:patch_size - overlap : h-patch_size+1;

cnt = 0;
NumberOfProccesedPatches = 0;

%%
Zeros_l = zeros(size(Dl_r));
A_l    = [ Dl_r      Zeros_l     Zeros_l ;...
           Zeros_l   Dl_g        Zeros_l ;...
           Zeros_l   Zeros_l     Dl_b    ];

Zeros_ltl = zeros(size(Dl_r'*Dl_r));

       
d1 = 0.5*[ Dl_r'*Dl_r  Zeros_ltl   Zeros_ltl ;...
           Zeros_ltl   Dl_g'*Dl_g  Zeros_ltl ;...
           Zeros_ltl   Zeros_ltl   Dl_b'*Dl_b];

% GradMatrix = [-2 -2 0; -2 0 2; 0 2 2]/6;
% GradMatrix = fspecial('laplacian') ;%生成一个2D 3*3滤波器
% GradOperator = MakeGradOperator(GradMatrix,patch_size^2);
% StS = GradOperator'* GradOperator;
%% Taking edges of dictionary atoms
% S_Dh = zeros(size(Dh));
% h_Laplacian = fspecial('laplacian') ;
% for ind =1:size(Dh,2)
%     Temp_Patch = reshape(Dh(:,ind), patch_size,patch_size);
%     Temp_Patch_Edge = imfilter(Temp_Patch,h_Laplacian,'same','conv');
%     S_Dh(:,ind) = reshape(Temp_Patch_Edge , patch_size^2,1);
% end
% S_Dh_r = S_Dh(:, 1                           : size(Dh_r,2));
% S_Dh_g = S_Dh(:, size(Dh_r,2)+1              : size(Dh_r,2)+size(Dh_g,2));
% S_Dh_b = S_Dh(:, size(Dh_r,2)+size(Dh_g,2)+1 : end);
% 
% Zeros_hth = zeros(size(Dh_r'*Dh_r));
%    
% d2 = 2*[ S_Dh_r'*S_Dh_r     -S_Dh_r'*S_Dh_g     Zeros_hth        ;...
%          Zeros_hth          S_Dh_g'*S_Dh_g      -S_Dh_g'*S_Dh_b  ;...
%         -S_Dh_b'*S_Dh_r     Zeros_hth           S_Dh_b'*S_Dh_b   ]; 
d2 = 0;   
%%
mNorm_R = zeros(length(gridy),length(gridx));
mNorm_G = zeros(length(gridy),length(gridx));
mNorm_B = zeros(length(gridy),length(gridx));

mMean_R = zeros(length(gridy),length(gridx));
mMean_G = zeros(length(gridy),length(gridx));
mMean_B = zeros(length(gridy),length(gridx));

mfNorm_R = zeros(length(gridy),length(gridx));
mfNorm_G = zeros(length(gridy),length(gridx));
mfNorm_B = zeros(length(gridy),length(gridx));

RHO_C = zeros(length(gridy),length(gridx));
Indicator = zeros(length(gridy),length(gridx));

y_R = zeros(length(gridy),length(gridx), size(lImfea_r,3)*patch_size^2);
y_G = zeros(length(gridy),length(gridx), size(lImfea_r,3)*patch_size^2);
y_B = zeros(length(gridy),length(gridx), size(lImfea_r,3)*patch_size^2);
y_L = zeros(length(gridy),length(gridx), 3*size(lImfea_r,3)*patch_size^2);

oldProgress= 0;
fprintf('Progress is : %f \n',oldProgress);
% loop to recover each low-resolution patch
for jj = 1:length(gridy),
    for ii = 1:length(gridx),
        newProgress = floor(100*cnt/(length(gridx)*length(gridy)))/100;
        if newProgress ~= oldProgress
            fprintf('Progress is : %f \n',newProgress);
        end
        oldProgress = newProgress;
         
        cnt = cnt+1;
        xx = gridx(ii);
        yy = gridy(jj);
%% 2.1 mean pixel value       
        mPatch_r = mIm_r(yy:yy+patch_size-1, xx:xx+patch_size-1);
        mMean_r = mean(mPatch_r(:));
        mPatch_r = mPatch_r(:) - mMean_r;
        mNorm_r = sqrt(sum(mPatch_r.^2));
        
        mNorm_R(yy,xx) = mNorm_r;
        
        mPatch_g = mIm_g(yy:yy+patch_size-1, xx:xx+patch_size-1);
        mMean_g = mean(mPatch_g(:));
        mPatch_g = mPatch_g(:) - mMean_g;
        mNorm_g = sqrt(sum(mPatch_g.^2));
        mStd_g = std(mPatch_g);
        
        mNorm_G(yy,xx) = mNorm_g;

        mPatch_b = mIm_b(yy:yy+patch_size-1, xx:xx+patch_size-1);
        mMean_b = mean(mPatch_b(:));
        mPatch_b = mPatch_b(:) - mMean_b;
        mNorm_b = sqrt(sum(mPatch_b.^2));
        
        mNorm_B(yy,xx) = mNorm_b;
        
        mPathc_y = im_m_ycbcr(yy:yy+patch_size-1, xx:xx+patch_size-1,1);
        mPathc_cb = im_m_ycbcr(yy:yy+patch_size-1, xx:xx+patch_size-1,2);
        mPathc_cr = im_m_ycbcr(yy:yy+patch_size-1, xx:xx+patch_size-1,3);
        
        mPatchFea_r = lImfea_r(yy:yy+patch_size-1, xx:xx+patch_size-1, :);   
        mPatchFea_r = mPatchFea_r(:);
        mfNorm_r = sqrt(sum(mPatchFea_r.^2));
        
        mfNorm_R(yy,xx) = mfNorm_r;
       
        if mfNorm_r > 1,
            y_r = mPatchFea_r./mfNorm_r;
        else
            y_r = mPatchFea_r;
        end
        
        y_R(yy,xx,:) = y_r;

        mPatchFea_g = lImfea_g(yy:yy+patch_size-1, xx:xx+patch_size-1, :);   
        mPatchFea_g = mPatchFea_g(:);
        mfNorm_g = sqrt(sum(mPatchFea_g.^2));
        
        mfNorm_G(yy,xx) = mfNorm_g;

        if mfNorm_g > 1,
            y_g = mPatchFea_g./mfNorm_g;
        else
            y_g = mPatchFea_g;
        end

        y_G(yy,xx,:) = y_g;

        mPatchFea_b = lImfea_b(yy:yy+patch_size-1, xx:xx+patch_size-1, :);   
        mPatchFea_b = mPatchFea_b(:);
        mfNorm_b = sqrt(sum(mPatchFea_b.^2));
        
        mfNorm_B(yy,xx) = mfNorm_b;
        
        if mfNorm_b > 1,
            y_b = mPatchFea_b./mfNorm_b;
        else
            y_b = mPatchFea_b;
        end
        
        y_B(yy,xx,:) = y_b;
         
        Y_l = [y_r y_g y_b];
        y_l = [y_r; y_g; y_b];
        
        y_L(:,:,1         :   end/3) = y_R;
        y_L(:,:,  end/3+1 : 2*end/3) = y_G;
        y_L(:,:,2*end/3+1 :   end  ) = y_B;
%%
%         ImageLength = size(mPathc_y,1)*size(mPathc_y,2) ;
%         
%         beta4 = var(im2double(mPathc_cb(:))) + var(im2double(mPathc_cr(:)));
%         rho_c = 0.001;
%         RHO_C (yy,xx) = rho_c;
%         if beta4>0.001
%             Indicator(yy,xx) = 1;
%         else
%             Indicator(yy,xx) = 0;
%         end
%         
%         if beta4>0.0015
%             D = d1+rho_c*d2;
% %             fprintf('C%.3f ',beta4)
%             NumberOfProccesedPatches = NumberOfProccesedPatches+1;
%             S_fista = FISTA_EdgeSimilarity(y_l,A_l, D ,lambda11);
%             S=S_fista;
%             
%         else
%% 2.2 sparse recovery

            A_r = Dl_r'*Dl_r ;
            A_g = Dl_g'*Dl_g ;
            A_b = Dl_b'*Dl_b ;
            
            b_r = - y_r'*Dl_r;
            b_g = - y_g'*Dl_g;
            b_b = - y_b'*Dl_b;
            
            w_r = L1QP_FeatureSign_yang(lambda1, A_r, b_r');
            w_g = L1QP_FeatureSign_yang(lambda1, A_g, b_g');
            w_b = L1QP_FeatureSign_yang(lambda1, A_b, b_b');
            
            S = [w_r ; w_g; w_b];
%         end
    

        
        if norm(S)>2
            fprintf(2,'\n Unexpected Code');
            w_r = S(1:end/3);
            w_g = S(end/3+1:2*end/3);
            w_b = S(2*end/3 +1 :end);
        end
        
 
%%         
        w_r = S(1:end/3);
        w_g = S(end/3+1:2*end/3);
        w_b = S(2*end/3 +1 :end);
        [w_r,w_g,w_b];
        
        % 2.3 generate the high resolution patch and scale the contrast
        
        Datatype = 'Line_expand';
        GraphConstructMethod = 'AdaptiveKNN_Global';
        
        Y        = Dho*S;
        Y        = Y';
        row      = 75;
        col      = 1;
        L        = HyperGraphCal_wu(Y,row,col,K,Datatype,GraphConstructMethod);
      
     
        [lm,ln]  = size(L);
        Ah       = lammda2 * L + 0.5 * eye(lm) ;
        bh       = -Dho*S;
        y_hh     = L1QP_FeatureSign_yang(0,Ah,bh);
        
        
%          hPatch_r = Dh_r*w_r;
         hPatch_r = y_hh(1:patch_size*patch_size, : );
        hPatch_r = lin_scale(hPatch_r, mNorm_r);
        
        hPatch_r = reshape(hPatch_r, [patch_size, patch_size]);
        hPatch_r = hPatch_r + mMean_r;
        
        hIm_r(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm_r(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch_r;
        cntMat_r(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat_r(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;
        %
%          hPatch_g = Dh_g*w_g;
         hPatch_g = y_hh(patch_size*patch_size+1 : 2*patch_size*patch_size , : );
        hPatch_g = lin_scale(hPatch_g, mNorm_g);
        
        hPatch_g = reshape(hPatch_g, [patch_size, patch_size]);
        hPatch_g = hPatch_g + mMean_g;
        
        hIm_g(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm_g(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch_g;
        cntMat_g(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat_g(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;
        %
%         hPatch_b = Dh_b*w_b;
        hPatch_b = y_hh(2*patch_size*patch_size+1 : end , : );
        hPatch_b = lin_scale(hPatch_b, mNorm_b);
        
        hPatch_b = reshape(hPatch_b, [patch_size, patch_size]);
        hPatch_b = hPatch_b + mMean_b;
        
        hIm_b(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm_b(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch_b;
        cntMat_b(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat_b(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;

        
    end
end

% fill in the empty with bicubic interpolation
idx_r = (cntMat_r < 1);
hIm_r(idx_r) = mIm_r(idx_r);

cntMat_r(idx_r) = 1;
hIm_r = hIm_r./cntMat_r;
hIm_r = uint8(hIm_r);

idx_g = (cntMat_g < 1);
hIm_g(idx_g) = mIm_g(idx_g);

cntMat_g(idx_g) = 1;
hIm_g = hIm_g./cntMat_g;
hIm_g = uint8(hIm_g);

idx_b = (cntMat_b < 1);
hIm_b(idx_b) = mIm_b(idx_b);

cntMat_b(idx_b) = 1;
hIm_b = hIm_b./cntMat_b;
hIm_b = uint8(hIm_b);

hIm(:,:,1) = hIm_r;
hIm(:,:,2) = hIm_g;
hIm(:,:,3) = hIm_b;
