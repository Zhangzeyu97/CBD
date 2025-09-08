import os
import logging
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np

parent_folder1 ='./results/'
task_list = ['case_1', 'case_2', 'case_3', 'case_4', 'case_5', 'case_6']
subfolder1 ='/task2a/test_latest/images/'
subtask = [['fake_A','real_A'],['fake_B','real_B'],['fake_C','real_C'],['fake_D','real_D'],['fake_E','real_E'],['fake_H','real_H']]

# Set up logging
log_file = os.path.join(parent_folder1, 'results.txt')
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)

# Add a StreamHandler to the logger to also print to console
console = logging.StreamHandler()
logging.getLogger().addHandler(console)

for m in range(len(task_list)):
    logging.info(task_list[m])
    for n in range(len(subtask)):
        folder1 = parent_folder1+task_list[m]+subfolder1+subtask[n][0]
        folder2 = parent_folder1+task_list[m]+subfolder1+subtask[n][1]

        try:
            files1 = os.listdir(folder1)
            files2 = os.listdir(folder2)
        except FileNotFoundError:
            logging.info(f"The folder {folder1} or {folder2} does not exist.")
            continue

        image_num = 500  # choose how many images to process
        count_ssim = 0
        count_psnr = 0
        for i in range(image_num):
            image1 = np.array(Image.open(os.path.join(folder1, files1[i])).convert('RGB'))
            image2 = np.array(Image.open(os.path.join(folder2, files2[i])).convert('RGB'))
            ssimval = ssim(image1, image2, multichannel=True, channel_axis=2)
            peaksnr = psnr(image1, image2)
            count_ssim += ssimval
            count_psnr += peaksnr

        count_ssim = count_ssim / image_num
        count_psnr = count_psnr / image_num

        logging.info(f'SSIM result: {count_ssim}, PSNR result: {count_psnr}')

logging.info(f"Finished all tasks. The results are stored in {log_file}.")
