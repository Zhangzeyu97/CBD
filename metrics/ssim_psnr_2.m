clear all;
% 
parent_folder1 ='/media/zeyu/108511CA108511CA/BID/results/';
task_list = {'case_1', 'case_2', 'case_3', 'case_4', 'case_5', 'case_6'};
% task_list = {'results'};

subfolder1 ='/task2a/test_latest/images/';
subtask = {{'fake_A','real_A'},{'fake_B','real_B'},{'fake_C','real_C'},{'fake_D','real_D'},{'fake_E','real_E'}};%{'fake_MixA','real_A'},


for m=1:length(task_list)
disp(task_list{m});
for n=1:length(subtask)


folder1 =strcat(parent_folder1,task_list{m},subfolder1,subtask{n}{1});
% disp(folder1)
folder2 =strcat(parent_folder1,task_list{m},subfolder1,subtask{n}{2});
% disp(folder2)

files1 = dir(folder1);
files2 = dir(folder2);

image_num=500; %choose how many images to process, 330.1795 /30300 for task I, 500 for task II.
count_ssim=0;
count_psnr=0;
err_count=0;    
for i=3:image_num+2
image1=uint8(imread(strcat(folder1,'/',files1(i).name)));
image2=uint8(imread(strcat(folder2,'/',files2(i).name)));
[ssimval,ssimmap]=ssim(image1,image2);
[peaksnr, snr] = psnr(image1,image2);
count_ssim = count_ssim + ssimval;
count_psnr = count_psnr + peaksnr;
end

count_ssim=count_ssim/image_num;
count_psnr=count_psnr/image_num;

fprintf('SSIM result: %f, PSNR result: %f\n', count_ssim, count_psnr);

end

end









    

