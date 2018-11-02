close all; clear all; clc;

%% Input image name
Name = 'face';
name_hr = [Name '.bmp'];

%ze
f = 'Data\Testing\face.bmp';

% set parameters
lambda1    = 0.1;                   % sparsity regularization
lammda2    = 0.001;                  % scaling factor, depending on the trained dictionary
maxIter    = 20;                   % if 0, do not use backprojection
DicSize    = 512;
patch_size = 5;
up_scale   = 3;
overlap    = 4;
K          = 20;

sampPerDeg = 23;
load displaySPD;
load SmithPokornyCones;
rgb2lms = cones'* displaySPD;
load displayGamma;
rgbWhite = [1 1 1];
whitepoint = rgbWhite * rgb2lms'
%% read ground truth image
im = imread(['Data/Testing/',name_hr]);
% im = imcrop(im);
img1{1} = im ;
img1 = modcrop(img1, up_scale);%图像的尺寸从[64,64,3]到[63,63,3]
low = resize(img1, 1/up_scale , 'bicubic');%imresize改变图像的大小，‘bicubic’双三次性插值,缩小了1/3，降采样得到低分辨率图像？
im_l = low{1};
disp(name_hr)
imwrite(im_l, strcat(Name,'_','or','.bmp'));

load('Dictionary/1yz20hp50_Color_D_20_100000_512_0.001_0.5_5_s3.mat');
maxDh=max(max(Dh));
maxDl=max(max(Dl));
minDh=min(min(Dh));
minDl=min(min(Dl));

% change color space, work on illuminance only
im_l_ycbcr = rgb2ycbcr(im_l);%低分辨率图像转换空间到 Y,Cb，Cr
im_l_y = im_l_ycbcr(:, :, 1);
im_l_cb = im_l_ycbcr(:, :, 2);
im_l_cr = im_l_ycbcr(:, :, 3);
%% HG image super-resolution based on sparse representation
im_l_r=im_l(:,:,1);
im_l_g=im_l(:,:,1);
im_l_b=im_l(:,:,1);
%% hypergraph
fprintf('-------------------------------------\n');
fprintf('-------------hypergraph scsr------------------------\n');
fprintf('-------------------------------------\n');

[im_h_hg] = MCcSR_hg(im_l, up_scale,  lambda1, lammda2, overlap, DicSize, patch_size, K);

[im_h1_r] = backprojection(im_h_hg(:,:,1), im_l(:,:,1), maxIter);
[im_h1_g] = backprojection(im_h_hg(:,:,2), im_l(:,:,2), maxIter);
[im_h1_b] = backprojection(im_h_hg(:,:,3), im_l(:,:,3), maxIter);

im_HR1_hg(:,:,1) = im_h1_r;
im_HR1_hg(:,:,2) = im_h1_g;
im_HR1_hg(:,:,3) = im_h1_b;

im_h_final_hg = (uint8(im_HR1_hg));

im_h_hg = shave(uint8(im_h_hg), [1 1] * up_scale);
im_h_final_hg = shave(uint8(im_h_final_hg), [1 1] * up_scale);

%% compute PSNR & SSIM

imwrite(im_h_final_hg, strcat(Name,'_','hg3','.bmp'));
im_shaved = shave(uint8(img1{1}), [1 1] * up_scale);

My_Full_rmseRGBhg = compute_rmseRGB(im_shaved, im_h_final_hg);
My_Full_SSIM_RGBhg(1) = ssim (im_shaved(:,:,1), im_h_final_hg(:,:,1)); My_Full_SSIM_RGBhg(2)=ssim(im_shaved(:,:,2), im_h_final_hg(:,:,2)); My_Full_SSIM_RGBhg(3)=ssim(im_shaved(:,:,3), im_h_final_hg(:,:,3))   ;

My_Full_psnrRGBhg = 20*log10(255/My_Full_rmseRGBhg);
ssim_hg=mean(My_Full_SSIM_RGBhg);

fprintf('-------------------------------------\n');
fprintf('RGB mse for MCcSR_hp: %f \n', My_Full_rmseRGBhg);
fprintf('-------------------------------------\n');
fprintf('RGB PSNR for MCcSR_hp: %f dB\n', My_Full_psnrRGBhg);
fprintf('-------------------------------------\n');
fprintf('RGB SSIM for MCcSR_hp: %.3f \n', ssim_hg);
ss=im_shaved-im_h_final_hg;
figure,imshow(ss);
%vif_hg
vif_hg1(1) = vifvec(im_shaved(:,:,1),im_h_final_hg(:,:,1));
vif_hg1(2) = vifvec(im_shaved(:,:,2),im_h_final_hg(:,:,2));
vif_hg1(3) = vifvec(im_shaved(:,:,3),im_h_final_hg(:,:,3));
vif_hg = mean(vif_hg1);

%% zeyde
fprintf('-------------------------------------\n');
fprintf('-------------  ze  ------------------------\n');
fprintf('-------------------------------------\n');
load('conf_Zeyde_512_finalx3.mat');
[img, imgCB, imgCR] = load_images({f}); 

img = modcrop(img, conf.scale^conf.level);
imgCB = modcrop(imgCB, conf.scale^conf.level);
imgCR = modcrop(imgCR, conf.scale^conf.level);

low = resize(img, 1/conf.scale^conf.level, conf.interpolate_kernel);

 lowCB = resize(imgCB, 1/conf.scale^conf.level, conf.interpolate_kernel);
 lowCR = resize(imgCR, 1/conf.scale^conf.level, conf.interpolate_kernel);
 
interpolatedCB = resize(lowCB, conf.scale^conf.level, conf.interpolate_kernel);    
interpolatedCR = resize(lowCR, conf.scale^conf.level, conf.interpolate_kernel);    
% conf.filenames = glob(input_dir, pattern); % Cell array
conf.desc = {'Original', 'Our algorithm'};
conf.results = {};
[res] = scaleup_Zeyde(conf, low);
re_ze=res{1};
re_ze = shave(uint8(re_ze * 255), conf.border * conf.scale);

re_zecb=interpolatedCB{1};
re_zecr=interpolatedCR{1};
re_zecb = shave(uint8(re_zecb * 255), conf.border * conf.scale);
re_zecr = shave(uint8(re_zecr * 255), conf.border * conf.scale);

rgbImg = cat(3, re_ze ,re_zecb,re_zecr);
im_h_final_ze = ycbcr2rgb(rgbImg);
im_h_final_ze = (uint8(im_h_final_ze));

imwrite(im_h_final_ze, strcat(Name,'_','ze3','.bmp'));

imwrite(im_shaved, strcat(Name,'_','origin','.bmp'));

My_Full_rmseRGBze = compute_rmseRGB(im_shaved, im_h_final_ze);
My_Full_SSIM_RGBze(1) = ssim (im_shaved(:,:,1), im_h_final_ze(:,:,1)); My_Full_SSIM_RGBze(2)=ssim(im_shaved(:,:,2), im_h_final_ze(:,:,2)); My_Full_SSIM_RGBze(3)=ssim(im_shaved(:,:,3), im_h_final_ze(:,:,3))   ;
My_Full_psnrRGBze = 20*log10(255/My_Full_rmseRGBze);
ssim_ze=mean(My_Full_SSIM_RGBze);

fprintf('-------------------------------------\n');
fprintf('RGB mse for ze: %f \n', My_Full_rmseRGBze);

fprintf('-------------------------------------\n');
fprintf('RGB PSNR for ze: %f dB\n', My_Full_psnrRGBze);
fprintf('-------------------------------------\n');
fprintf('RGB SSIM for ze: %.3f \n', ssim_ze);

%VIF_ZE
vif_ze1(1) = vifvec(im_shaved(:,:,1),im_h_final_ze(:,:,1));
vif_ze1(2) = vifvec(im_shaved(:,:,2),im_h_final_ze(:,:,2));
vif_ze1(3) = vifvec(im_shaved(:,:,3),im_h_final_ze(:,:,3));
vif_ze = mean(vif_ze1);

%% ANR
fprintf('-------------------------------------\n');
fprintf('-------------  anr  ------------------------\n');
fprintf('-------------------------------------\n');

conf.ProjM = inv(conf.dict_lores'*conf.dict_lores+0.01*eye(size(conf.dict_lores,2)))*conf.dict_lores';    
conf.PP = (1+0.01)*conf.dict_hires*conf.ProjM;
    conf.points = [1:1:size(conf.dict_lores,2)];
    
    conf.pointslo = conf.dict_lores(:,conf.points);
    conf.pointsloPCA = conf.pointslo'*conf.V_pca';
    conf.PPs = [];   
    if  size(conf.dict_lores,2) < 40
        clustersz = size(conf.dict_lores,2);
    else
        clustersz = 40;
    end
    D = abs(conf.pointslo'*conf.dict_lores);    
    
    for i = 1:length(conf.points)
        [vals idx] = sort(D(i,:), 'descend');
        if (clustersz >= size(conf.dict_lores,2)/2)
            conf.PPs{i} = conf.PP;
        else
            Lo = conf.dict_lores(:, idx(1:clustersz));        
            conf.PPs{i} = 1.01*conf.dict_hires(:,idx(1:clustersz))*inv(Lo'*Lo+0.01*eye(size(Lo,2)))*Lo';    
        end
    end    
    
    ANR_PPs = conf.PPs; % store the ANR regressors
    
conf.PPs = ANR_PPs;
conf.desc = {'Original', 'Our algorithm'};
conf.results = {};

re_ANR = scaleup_ANR(conf, low);

re_anr=re_ANR{1};
re_anr = shave(uint8(re_anr * 255), conf.border * conf.scale);

re_anrcb=interpolatedCB{1};
re_anrcr=interpolatedCR{1};
re_anrcb = shave(uint8(re_anrcb * 255), conf.border * conf.scale);
re_anrcr = shave(uint8(re_anrcr * 255), conf.border * conf.scale);

rgbImganr = cat(3, re_anr ,re_anrcb,re_anrcr);
im_h_final_anr = ycbcr2rgb(rgbImganr);
im_h_final_anr = (uint8(im_h_final_anr));

imwrite(im_h_final_anr, strcat(Name,'_','ANR3','.bmp'));

My_Full_rmseRGBanr = compute_rmseRGB(im_shaved, im_h_final_anr);
My_Full_SSIM_RGBanr(1) = ssim (im_shaved(:,:,1), im_h_final_ze(:,:,1)); My_Full_SSIM_RGBanr(2)=ssim(im_shaved(:,:,2), im_h_final_anr(:,:,2)); My_Full_SSIM_RGBanr(3)=ssim(im_shaved(:,:,3), im_h_final_anr(:,:,3))   ;
My_Full_psnrRGBanr = 20*log10(255/My_Full_rmseRGBanr);
ssim_anr=mean(My_Full_SSIM_RGBanr);
fprintf('-------------------------------------\n');
fprintf('RGB mse for ANR: %f \n', My_Full_rmseRGBanr);
fprintf('-------------------------------------\n');
fprintf('RGB PSNR for ANR: %f dB\n', My_Full_psnrRGBanr);
fprintf('-------------------------------------\n');
fprintf('RGB SSIM for ANR: %.3f \n', ssim_anr);
%VIF_ANR
vif_anr1(1) = vifvec(im_shaved(:,:,1),im_h_final_anr(:,:,1));
vif_anr1(2) = vifvec(im_shaved(:,:,2),im_h_final_anr(:,:,2));
vif_anr1(3) = vifvec(im_shaved(:,:,3),im_h_final_anr(:,:,3));
vif_anr = mean(vif_anr1);

%% bicubic interpolation for reference
fprintf('-------------------------------------\n');
fprintf('-------------  bb  ------------------------\n');
fprintf('-------------------------------------\n');
[nrow, ncol,~] = size(img1{1});
im_b = imresize(im_l, [nrow, ncol], 'bicubic');

% Shaving the results and the input
im_b = shave(uint8(im_b), [1 1] * up_scale);

imwrite(im_b, strcat(Name,'_','bb3','.bmp'));

bb_rmse_RGB = compute_rmseRGB(im_shaved, im_b);
bb_SSIM_RGB(1) = ssim(im_shaved(:,:,1), im_b(:,:,1)); bb_SSIM_RGB(2)=ssim(im_shaved(:,:,2), im_b(:,:,2)); bb_SSIM_RGB(3)=ssim(im_shaved(:,:,3), im_b(:,:,3))   ;
bb_psnrRGB = 20*log10(255/bb_rmse_RGB);
ssim_bb=mean(bb_SSIM_RGB);
fprintf('-------------------------------------\n');
fprintf('RGB mse for Bicubic Interpolation: %f \n', bb_rmse_RGB);
fprintf('-------------------------------------\n');
fprintf('RGB PSNR for Bicubic Interpolation: %f dB\n', bb_psnrRGB);
fprintf('-------------------------------------\n');
fprintf('RGB SSIM for Bicubic Interpolation: %.3f  \n', mean(bb_SSIM_RGB));
%VIF_bb
vif_bb1(1) = vifvec(im_shaved(:,:,1),im_b(:,:,1));
vif_bb1(2) = vifvec(im_shaved(:,:,2),im_b(:,:,2));
vif_bb1(3) = vifvec(im_shaved(:,:,3),im_b(:,:,3));
vif_bb = mean(vif_bb1);


%% image super-resolution based on sparse representation
fprintf('-------------------------------------\n');
fprintf('-------------  mccsr  ------------------------\n');
fprintf('-------------------------------------\n');

Dl = [];
Dh = [];

lambda = 0.1;%0.1;                   % sparsity regularization

[im_h_mc1] = MCcSR_EdgeSimilarityLearnedDicSimultaneous(im_l, up_scale, lambda, overlap, DicSize);
% upscale the chrominance simply by "bicubic"  
[nrow, ncol,~] = size(im_h_mc1);
[im_h1_r] = backprojection(im_h_mc1(:,:,1), im_l(:,:,1), maxIter);
[im_h1_g] = backprojection(im_h_mc1(:,:,2), im_l(:,:,2), maxIter);
[im_h1_b] = backprojection(im_h_mc1(:,:,3), im_l(:,:,3), maxIter);

im_HR1mc(:,:,1) = im_h1_r;
im_HR1mc(:,:,2) = im_h1_g;
im_HR1mc(:,:,3) = im_h1_b;

im_h_final_mc = (uint8(im_HR1mc));
im_h_final_mc = shave(uint8(im_h_final_mc), [1 1] * up_scale);

imwrite(im_h_final_mc, strcat(Name,'_','mccsr3','.bmp'));

My_Full_rmseRGBmc = compute_rmseRGB(im_shaved, im_h_final_mc);
My_Full_SSIM_RGBmc(1) = ssim (im_shaved(:,:,1), im_h_final_mc(:,:,1)); My_Full_SSIM_RGBmc(2)=ssim(im_shaved(:,:,2), im_h_final_mc(:,:,2)); My_Full_SSIM_RGBmc(3)=ssim(im_shaved(:,:,3), im_h_final_mc(:,:,3))   ;


My_Full_psnrRGBmc = 20*log10(255/My_Full_rmseRGBmc);
ssim_mc=mean(My_Full_SSIM_RGBmc);

fprintf('-------------------------------------\n');
fprintf('RGB mse for MCcSR: %f \n', My_Full_rmseRGBmc);
fprintf('-------------------------------------\n');
fprintf('RGB PSNR for MCcSR: %f dB\n', My_Full_psnrRGBmc);
fprintf('-------------------------------------\n');
fprintf('RGB SSIM for MCcSR: %.3f \n', mean(My_Full_SSIM_RGBmc));
%vif_mc
vif_mc1(1) = vifvec(im_shaved(:,:,1),im_h_final_mc(:,:,1));
vif_mc1(2) = vifvec(im_shaved(:,:,2),im_h_final_mc(:,:,2));
vif_mc1(3) = vifvec(im_shaved(:,:,3),im_h_final_mc(:,:,3));
vif_mc = mean(vif_mc1);
%% scielab  mc
% Convert the RGB data to LMS (or XYZ if you like).
% img6 = [im_h_final_mc(:,:,1) im_h_final_mc(:,:,2) im_h_final_mc(:,:,3)];
% imgRGB6 = dac2rgb(img6,gammaTable);
% img2LMS_mc = changeColorSpace(imgRGB6,rgb2lms);
% %   Run the scielab function.
% errorImage_mc = scielab(sampPerDeg, img1LMS, img2LMS_mc, whitepoint, imageformat);
% 
% figure(5)
% hist(errorImage_mc(:))
% error_mc=sum(errorImage_mc(:))   % We think this is 173
% % Look at the spatial distribution of the errors.
% errorTruncated_mc = min(128*(errorImage_mc/10),128*ones(size(errorImage_mc)));
% colormap(gray(128));
% image(errorTruncated_mc); axis image;
% % edgeImage = 129 * double(edge(im_shaved(:,:,1),'prewitt'));
% comparison5 = max(edgeImage,errorTruncated_mc);
% mp5 = [gray(127)];
% colormap(mp5),image(comparison5),title('mc');

%% SRSR
fprintf('-------------------------------------\n');
fprintf('-------------  SR  ------------------------\n');
fprintf('-------------------------------------\n');
Dl = [];
Dh = [];
load('Dictionary/D_100000_512_0.1_5_s3.mat');

% image super-resolution based on sparse representation
[im_h_y] = ScSR(im_l_y, up_scale, Dh, Dl, lambda1, overlap);
[im_h_y] = backprojection(im_h_y, im_l_y, maxIter);

% upscale the chrominance simply by "bicubic" 
[nrow, ncol] = size(im_h_y);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');
% upscale the chrominance simply by "bicubic" 
im_h_ycbcr = zeros([nrow, ncol, 3]);
im_h_ycbcr(:, :, 1) = im_h_y;
im_h_ycbcr(:, :, 2) = im_h_cb;
im_h_ycbcr(:, :, 3) = im_h_cr;
im_h_final_sr = ycbcr2rgb(uint8(im_h_ycbcr));

% im_h = shave(uint8(im_h), [1 1] * up_scale);
im_h_final_sr = shave(uint8(im_h_final_sr), [1 1] * up_scale);
imwrite(im_h_final_sr, strcat(Name,'_','sr3','.bmp'));

im_shaved_ycbcr = rgb2ycbcr(im_shaved);

%% compute PSNR & SSIM
% im_h_ycbcr = rgb2ycbcr(im_h);
My_Full_rmseRGBsr = compute_rmseRGB(im_shaved, im_h_final_sr);
My_Full_SSIM_RGBsr(1) = ssim (im_shaved(:,:,1), im_h_final_sr(:,:,1)); My_Full_SSIM_RGBsr(2)=ssim(im_shaved(:,:,2), im_h_final_sr(:,:,2)); My_Full_SSIM_RGBsr(3)=ssim(im_shaved(:,:,3), im_h_final_sr(:,:,3))   ;

My_Full_psnrRGBsr = 20*log10(255/My_Full_rmseRGBsr);
ssim_sr=mean(My_Full_SSIM_RGBsr);
fprintf('-------------------------------------\n');
fprintf('RGB mse for MCcSR_hp: %f dB\n', My_Full_rmseRGBsr);
fprintf('-------------------------------------\n');
fprintf('RGB PSNR for SR: %f dB\n', My_Full_psnrRGBsr);
fprintf('-------------------------------------\n');
fprintf('RGB SSIM for SR: %.3f \n', mean(My_Full_SSIM_RGBsr));


%vif_SR
vif_sr1(1) = vifvec(im_shaved(:,:,1),im_h_final_sr(:,:,1));
vif_sr1(2) = vifvec(im_shaved(:,:,2),im_h_final_sr(:,:,2));
vif_sr1(3) = vifvec(im_shaved(:,:,3),im_h_final_sr(:,:,3));
vif_sr = mean(vif_sr1);

