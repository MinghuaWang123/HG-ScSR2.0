function [Xh, Xl] = rnd_smp_patch(img_path, type, patch_size, num_patch, upscale)

img_dir = dir(fullfile(img_path, type));%fullfile���ɵ�ַ�ַ���;dir�������ָ���ļ����µ��������ļ��к��ļ�,���������һ��Ϊ�ļ��ṹ��������

Xhr1 = [];
Xlr1 = [];

Xhg1 = [];
Xlg1 = [];

Xhb1 = [];
Xlb1 = [];

img_num = length(img_dir);%69��ͼ
nper_img = zeros(1, img_num);

for ii = 1:length(img_dir),
    im = imread(fullfile(img_path, img_dir(ii).name));
    nper_img(ii) = prod(size(im));%B = prod(A) ��A����ͬά��Ԫ�صĳ˻����ص�����B��
end

nper_img = floor(nper_img*num_patch/sum(nper_img));

for ii = 1:img_num,
    patch_num = nper_img(ii);
    im = imread(fullfile(img_path, img_dir(ii).name));
    
    im_r=im(:,:,1);
    [Hr, Lr] = sample_patches(im_r, patch_size, patch_num, upscale);%��ȡLRͼ��Ĳ�ͬ����
    [m,n]=size(Hr);
    Xhr1 = [Xhr1, Hr];
    Xlr1 = [Xlr1, Lr];
    
    im_g=im(:,:,2);
    [Hg, Lg] = sample_patches(im_g, patch_size, patch_num, upscale);%��ȡLRͼ��Ĳ�ͬ����
    Xhg1 = [Xhg1, Hg];
    Xlg1 = [Xlg1, Lg]; 
    
    im_b=im(:,:,3);
    [Hb, Lb] = sample_patches(im_b, patch_size, patch_num, upscale);%��ȡLRͼ��Ĳ�ͬ����
    Xhb1 = [Xhb1, Hb];
    Xlb1 = [Xlb1, Lb];
    
end
[hm,hn]=size(Xhr1);
Xh=[Xhr1;Xhg1;Xhb1];
Xl=[Xlr1;Xlg1;Xlb1];
patch_path = ['Training/rnd_patches_' num2str(patch_size) '_' num2str(num_patch) '_s' num2str(upscale) '.mat'];
save(patch_path, 'Xh', 'Xl');