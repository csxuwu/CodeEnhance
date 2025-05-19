import os

from PIL import Image, ImageStat
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd


def brightness1( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]


def pd_toExcel(ids, data, fileName): 

   dfData = {  
      'ids': ids,
      'illu_avg': data,
   }
   df = pd.DataFrame(dfData)  
   df.to_excel(fileName, index=False)  
   print('ending')

def pd_toExcel2(data, fileName):  

   ids = []
   class_ = []
   avg_illu = []

   for i in range(len(data)):
      ids.append(data[i]['id'])
      class_.append(data[i]['class'])
      avg_illu.append(data[i]['avg_illu'])

   dfData = { 
      'ids': ids,
      'class': class_,
      'avg_illu': avg_illu,
   }
   df = pd.DataFrame(dfData)  
   df.to_excel(fileName, index=False) 
   print('ending')

if __name__=='__main__':

   

   # name = 'AGLLNet_noise'
   # # file_path = f'G:\Dataset\LL_Set\{name}'
   # file_path = r'/home/wuxu/datasets/LL/AGLLNet/train_lowlight/'

   file_dict = {
   'AGLLNet_noise': r'/home/wuxu/datasets/LL/AGLLNet/train_lowlight/',
   'AGLLNet_without_noise': r'/home/wuxu/datasets/LL/AGLLNet/train_dark/',
   'FiveK': r'/home/wuxu/datasets/LL/FiveK/train/input',
   'LSRW_Huawei': r'/home/wuxu/datasets/LL/LSRW/Training data/Huawei/low',
   'LSRW_Nikon': r'/home/wuxu/datasets/LL/LSRW/Training data/Nikon/low',
   }
   type_ops = '' 
   sample_num = 1000

   for k in file_dict:

      illumination_avg = [0] * 256     
      illumination_avg_list = []       
      ids = []
      for i in range(len(illumination_avg)):
         ids.append(i)

      name = k
      file_path = file_dict[k]

      img_list = glob(os.path.join(file_path, '*.*'))
      avg_illu_file = 0.0
      illu_avg_file = []

      j = 1


      if type_ops == 'full_mean':

         for img_path in tqdm(img_list):
            illu_avg = brightness1(img_path)
            illumination_avg_list.append(illu_avg)

            # if len(img_list) > 9999:
            #    if j % 100 == 0:
            #       illumination_avg_list.append(illu_avg)
            # else:
            #    illumination_avg_list.append(illu_avg)
            illu_avg = int(illu_avg)
            illumination_avg[illu_avg] += 1
            avg_illu_file += illu_avg
            j += 1


      else:

         index = np.random.randint(0, len(img_list), sample_num)
         for i in tqdm(index):
            img_path = img_list[i]

            illu_avg = brightness1(img_path)
            illumination_avg_list.append(illu_avg)

            illu_avg = int(illu_avg)
            illumination_avg[illu_avg] += 1
            avg_illu_file += illu_avg
            j += 1

      if type_ops == 'full_mean':
         avg_illu_file = avg_illu_file / len(img_list)
      else:
         avg_illu_file = avg_illu_file / sample_num
      print(f'avg illu of {k}: {avg_illu_file}')


      ids2 = []
      for i in range(len(illumination_avg_list)):
         ids2.append(i)

      fileName = f'/home/wuxu/datasets/LL/illumination_AVG_numbers_{name}_{sample_num}.xlsx'
      fileName2 = f'/home/wuxu/datasets/LL/illumination_AVG_{name}_{sample_num}.xlsx'
      pd_toExcel(ids=ids, data=illumination_avg,fileName=fileName)       
      pd_toExcel(ids=ids2, data=illumination_avg_list,fileName=fileName2) 










