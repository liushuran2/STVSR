import argparse
import yaml
import math
import matplotlib.pyplot as plt
from scipy.stats import laplace
from tqdm import tqdm
from PIL import Image
import warnings
from skimage.metrics import structural_similarity as ssim
warnings.filterwarnings("ignore")
f = open("config.yaml")
config = yaml.load(f, Loader=yaml.FullLoader)
yips = 0.05
import numpy as np
from torch.utils.data import DataLoader
from TestDataset import TestDataset
from  diagram import reliability_diagram,interval_confidence
import models
import torch
import loss

num_dropout_ensembles = 8
checkpt=torch.load('checkpt/exp/checkptbest.pt')
patch_size=512
factor = 3

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def calc_psnr(sr, hr, scale=3, rgb_range=255, dataset=None):
    hr = hr.squeeze(0)
    hr = hr.squeeze(0)
    hr = hr.cpu().detach().numpy()
    diff = (sr - hr)
    mse = np.mean((diff) ** 2)
    return -10 * math.log10(mse)
def calc_ssim(sr,hr):
    hr = hr.squeeze(0)
    hr = hr.squeeze(0)
    hr = hr.cpu().detach().numpy()
    return ssim(sr,hr)
def laplace_cdf(x, mu_list, b_list):
    cdf = 0.5 * np.sign(x - mu_list) * (1 - np.exp(-np.abs(x - mu_list) / b_list))
    return np.mean(cdf, axis=0)

model = torch.nn.DataParallel(models.mana(config,is_training=False)).cuda()
model.module.load_state_dict(checkpt)

loss_fn = loss.lossfun()

test_dataset = TestDataset('Lifeact_test_step2_VSR_gtnorm.h5', patch_size=config['patch_size'], scale=2)
dataloader = DataLoader(dataset=test_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config['num_workers'],
                        pin_memory=True)
count = 0

with torch.no_grad():
    model.eval()
    enable_dropout(model)
    with tqdm(dataloader, desc="Training Model") as tepoch:
        psnr_list=[]
        ssim_list=[]
        for inp, gt in tepoch:
            mean = np.ndarray((num_dropout_ensembles, factor*patch_size, factor*patch_size)) #这里要改
            data_uncertainty = np.ndarray((num_dropout_ensembles, factor*patch_size, factor*patch_size))
            count += 1
            inp = inp.float().cuda()
            gt = gt.float().cuda()
            for i in range(num_dropout_ensembles):
                oup,qloss = model(inp)
                SR_y = np.flip(oup[0, 0:1, :, :].permute(1, 2, 0).data.cpu().numpy(),2)
                SR_y = SR_y.astype(np.float32)
                SR_y = SR_y.squeeze(2)
                std_y = np.flip(oup[0, 1:2, :, :].permute(1, 2, 0).data.cpu().numpy(),2)
                std_y = std_y.astype(np.float32)
                std_y = std_y.squeeze(2)
                mean[i, :, :] = SR_y
                data_uncertainty[i, :, :] = std_y
        
            bin_confidence, bin_correct ,bin_total= reliability_diagram(gt.data.cpu().numpy(), mean, data_uncertainty, yips)
            bin_correct = np.array(bin_correct)
            bin_confidence = np.array(bin_confidence)
            bin_total = np.array(bin_total)
            if count == 1:
                bin_confidences = bin_confidence
                bin_corrects = bin_correct
                bin_totals = bin_total
            else:
                bin_confidences += bin_confidence
                bin_corrects += bin_correct
                bin_totals += bin_total

            SR_result = np.mean(mean, axis=0)
            data_uncertainty_result = np.mean(data_uncertainty, axis=0)
            model_uncertainty_result = np.std(mean, axis=0)

            psnr_list.append(calc_psnr(SR_result, gt))
            ssim_list.append(calc_ssim(SR_result, gt))

            SR_result = Image.fromarray(SR_result)
            SR_result.save('result/super_res{}.tif'.format(str(count)))

            confidence = interval_confidence(mean, data_uncertainty, num_dropout_ensembles, yips)
            confidence = Image.fromarray(confidence)
            confidence.save('result/confidence{}.tif'.format(str(count)))


            data_uncertainty_result = Image.fromarray(data_uncertainty_result)
            data_uncertainty_result.save('result/datauncertain_res{}.tif'.format(str(count)))

            model_uncertainty_result = Image.fromarray(model_uncertainty_result)
            model_uncertainty_result.save('result/modeluncertain_res{}.tif'.format(str(count)))


non_zero = bin_totals.nonzero()
non_zero = np.where(bin_totals > 100)
bin_confidences = bin_confidences[non_zero] / bin_totals[non_zero]
bin_corrects = bin_corrects[non_zero] / bin_totals[non_zero]


fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', label='Ideal')
ax.plot(bin_confidences, bin_corrects, marker='o', label='Model')
ax.set_xlabel('Average Confidence')
ax.set_ylabel('Accuracy')
ax.set_title('Reliability Diagram')
ax.legend()
plt.show()
plt.savefig('fig.png')