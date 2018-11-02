function [Xh, Xl] = rnd_smp_patch(img_path, type, patch_size, num_patch, upscale)

img_dir = dir(fullfile(img_path, type));%fullfile构成地址字符串;dir函数获得指定文件夹下的所有子文件夹和文件,并存放在在一种为文件结构体数组中

Xhr1 = [];
Xlr1 = [];

Xhg1 = [];
Xlg1 = [];

Xhb1 = [];
Xlb1 = [];

img_num = length(img_dir);%69张图
nper_img = zeros(1, img_num);

for ii = 1:length(img_dir),
    im = imread(fullfile(img_path, img_dir(ii).name));
    nper_img(ii) = prod(size(im));%B = prod(A) 将A矩阵不同维的元素的乘积返回到矩阵B。
end

nper_img = floor(nper_img*num_patch/sum(nper_img));

for ii = 1:img_num,
    patch_num = nper_img(ii);
    im = imread(fullfile(img_path, img_dir(ii).name));
    
    im_r=im(:,:,1);
    [Hr, Lr] = sample_patches(im_r, patch_size, patch_num, upscale);%提取LR图像的不同特征
    [m,n]=size(Hr);
    Xhr1 = [Xhr1, Hr];
    Xlr1 = [Xlr1, Lr];
    
    im_g=im(:,:,2);
    [Hg, Lg] = sample_patches(im_g, patch_size, patch_num, upscale);%提取LR图像的不同特征
    Xhg1 = [Xhg1, Hg];
    Xlg1 = [Xlg1, Lg]; 
    
    im_b=im(:,:,3);
    [Hb, Lb] = sample_patches(im_b, patch_size, patch_num, upscale);%提取LR图像的不同特征
    Xhb1 = [Xhb1, Hb];
    Xlb1 = [Xlb1, Lb];
    
end
[hm,hn]=size(Xhr1);
Xh=[Xhr1;Xhg1;Xhb1];
Xl=[Xlr1;Xlg1;Xlb1];
patch_path = ['Training/rnd_patches_' num2str(patch_size) '_' num2str(num_patch) '_s' num2str(upscale) '.mat'];
save(patch_path, 'Xh', 'Xl');