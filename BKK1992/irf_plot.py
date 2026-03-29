#%% 导入包
import mat73
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#%% 读取数据
d_mat = mat73.loadmat("~/dynare_irfs_v73.mat")




#%% 获取稳态字典
steady_state = np.asarray(d_mat["steady_state"]).squeeze()
endo_names_raw = np.asarray(d_mat["endo_names"]).squeeze()
endo_names = [str(x).strip() for x in endo_names_raw.tolist()]
ss_dict = {name: float(steady_state[i]) for i, name in enumerate(endo_names)}





#%% 获取irf取值
irfs = d_mat['irfs']
ifrs_df = pd.DataFrame(irfs)





#%% 本国对本国冲击
irf_self = ifrs_df.loc[:, ifrs_df.columns.str.endswith("_E_H")].copy()

# 获取产出稳态
y_ss = ss_dict['y_h']

# 计算相对产出的偏离程度
irf_self = 100.0 * irf_self / y_ss  
irf_self.columns = irf_self.columns.str.replace(r"_E_H$", "", regex=True)  



#%% 本国对别国冲击
irf_cross = ifrs_df.loc[:, ifrs_df.columns.str.endswith("_E_F")].copy()

# 获取产出稳态
y_ss = ss_dict['y_f']

# 计算相对产出的偏离程度
irf_cross = 100.0 * irf_cross / y_ss  
irf_cross.columns = irf_cross.columns.str.replace(r"_E_F$", "", regex=True)  




#%% 绘图
df_list = [irf_self,irf_cross]
for df in df_list:
    T = 20
    plot_df = df.iloc[:T, :].copy()
    plot_df.index = np.arange(1, T+1)

    # 加入初始偏离为0
    plot_df = pd.concat([pd.DataFrame(0, index=[0], columns=plot_df.columns), plot_df]).sort_index()

    plt.figure(figsize=(9, 5))
    for col in plot_df.columns:
        plt.plot(plot_df.index, plot_df[col].values, label=col)

    plt.axhline(0)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.show()


