"""
将所有数据从原始文件中读取，降清晰度，并存入单一npy文件
"""

from scipy import interpolate
import numpy as np
import netCDF4 as nc
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

soil_type_temp = np.load('D:\code\ERA5npydata\soil_type.npy')
soil_type_temp = soil_type_temp[0]
a,b = (np.where(soil_type_temp== 6))
soil_type_temp[a,b] = 2

a,b = (np.where(soil_type_temp== 7))
soil_type_temp[a,b] = 3


soil_type_temp=np.flipud(soil_type_temp)

n = soil_type_temp.shape[0]
n1 = soil_type_temp.shape[1]



lim = np.arange(0, 6, 1)
x = np.linspace(-6, 6, n1)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)

predictive_model_name = "soil type"
plt.contourf(X, Y, soil_type_temp, lim)
plt.colorbar(orientation='horizontal')
plt.title(predictive_model_name)
plt.show()


np.save('D:\code\ERA5npydata\soil_type1.npy', soil_type_temp)
