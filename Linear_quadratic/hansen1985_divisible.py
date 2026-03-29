
#%%----------------------------------------------------------------------
#--------------------导入包--------------------
#----------------------------------------------------------------------
import numpy as np
from scipy.optimize import root
import sympy as sp
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.linalg import solve_discrete_are







#%%----------------------------------------------------------------------
#--------------------设置参数--------------------
#----------------------------------------------------------------------
theta = 0.36
delta = 0.025
beta = 0.99
A = 2
gamma = 0.95
sigma_eps = 0.00712
sigmasq_eps = sigma_eps**2





#%%----------------------------------------------------------------------
#--------------------求解稳态--------------------
#----------------------------------------------------------------------
rho = 1/beta - 1
h_ss = (1-theta)*(rho+delta) / (A*(rho+delta-theta*delta) + (1-theta)*(rho+delta))
k_ss = h_ss * ((rho+delta)/theta)**(1/(theta-1))
y_ss = k_ss**theta * h_ss**(1-theta)
inv_ss = delta * k_ss
c_ss = y_ss - inv_ss
z_ss = 0
lambda_ss = np.exp(z_ss)



#%%----------------------------------------------------------------------
#--------------------稳态附近二阶展开--------------------
#----------------------------------------------------------------------

# 定义变量
k, z, kp, h = sp.symbols('k, z, kp, h', real=True)
theta_s, delta_s, A_s = sp.symbols('theta, delta, A',real=True)


# 函数表达式（最小化负效用）
y = sp.exp(z) * (k**theta_s) * (h**(1-theta_s))
c = y + (1-delta_s)*k - kp
u = -(sp.log(c) + A_s*sp.log(1-h))

# 固定变量顺序
vars_ = sp.Matrix([k,z,kp,h])


# 梯度和Hessian矩阵(参数与变量的表达式)
grad_expr = sp.Matrix([sp.diff(u,v) for v in vars_])
hess_expr = sp.hessian(u,vars_)


# 带入参数值
subs_par = {theta_s:theta,
            delta_s:delta,
            A_s:A}
grad_expr = grad_expr.subs(subs_par)
hess_expr = hess_expr.subs(subs_par)


# 编译为函数表达式
grad_fun = sp.lambdify((k,z,kp,h),grad_expr,modules='numpy')
hess_fun = sp.lambdify((k,z,kp,h),hess_expr,modules='numpy')
u_fun = sp.lambdify((k, z, kp, h), u.subs(subs_par), modules='numpy')


# 带入稳态值
kp_ss = k_ss

grad_ss = np.array(grad_fun(k_ss,z_ss,kp_ss,h_ss), dtype=float).reshape(-1)
hess_ss = np.array(hess_fun(k_ss, z_ss, kp_ss, h_ss), dtype=float)
u_ss = float(u_fun(k_ss, z_ss, kp_ss, h_ss))






#%%----------------------------------------------------------------------
#--------------------水平值对应的M矩阵--------------------
#----------------------------------------------------------------------

vbar = np.array([k_ss, z_ss, kp_ss, h_ss], dtype=float)
S = np.array([
    [-vbar[0], 1.0, 0.0, 0.0, 0.0],
    [-vbar[1], 0.0, 1.0, 0.0, 0.0],
    [-vbar[2], 0.0, 0.0, 1.0, 0.0],
    [-vbar[3], 0.0, 0.0, 0.0, 1.0],
], dtype=float)

g = grad_ss.reshape(4, 1)      # (4,1)
H = hess_ss


J = 0.5 * (S.T @ H @ S)     # (5,5)
b = (S.T @ g).reshape(5, 1)



e1 = np.zeros((5,1))
e1[0,0] = 1.0

M = J + 0.5*(e1 @ b.T + b @ e1.T) + u_ss*(e1 @ e1.T)


# 修正对称性
M = 0.5*(M + M.T)


# 分块
n = 3
m = 2
R = M[:n, :n]
W = M[n:, :n]
Q = M[n:, n:]
print("eig(Q) after flip =", np.linalg.eigvals(Q))






#%%----------------------------------------------------------------------
#--------------------线性约束--------------------
#----------------------------------------------------------------------
A_mat = np.array([
    [1.0, 0.0,   0.0],
    [0.0, 0.0,   0.0],
    [0.0, 0.0, gamma],
], dtype=float)

B_mat = np.array([
    [0.0, 0.0],
    [1.0, 0.0],   
    [0.0, 0.0],
], dtype=float)







#%%----------------------------------------------------------------------
#--------------------Riccati 方程直接获取稳定解--------------------
#----------------------------------------------------------------------

# 通过A/B矩阵吸收beta系数
Ab = np.sqrt(beta) * A_mat
Bb = np.sqrt(beta) * B_mat

# 求解P矩阵稳定解
P = solve_discrete_are(Ab, Bb, R, Q, s=W.T)

# 计算政策函数
G = Q + Bb.T @ P @ Bb          # = Q_c + beta B'PB
K = W + Bb.T @ P @ Ab          # = W_c + beta B'PA
F = -np.linalg.solve(G, K)

print("P=\n", P)
print("F=\n", F)
print("eig(G) =", np.linalg.eigvalsh(0.5*(G+G.T)))








#%%----------------------------------------------------------------------
#--------------------经济体的运动方程--------------------
#----------------------------------------------------------------------
'''
y = F * x
x(+1) = A * x + B * y + C * eps = (A+BF)*x + C*eps
'''
policy_mat = F
trans_mat = A_mat + B_mat @ policy_mat
shock_mat = np.array([[0.0],[0.0],[1.0]])





#%%----------------------------------------------------------------------
#--------------------simulation--------------------
#----------------------------------------------------------------------
# 随机种子
rng = np.random.default_rng(123)   


# 模拟参数
T = 115       
burn = 100      
rep_num = 100         



# 稳态时的状态变量
x_ss = np.array([1.0, k_ss, z_ss], dtype=float)



# 初始化路径储存
x_path = np.zeros((rep_num, T+1, n))
y_path = np.zeros((rep_num, T, m))


# 初始化经济变量
k_series   = np.zeros((rep_num, T))
z_series   = np.zeros((rep_num, T))
h_series   = np.zeros((rep_num, T))
kp_series  = np.zeros((rep_num, T))
y_series   = np.zeros((rep_num, T))   # output
c_series   = np.zeros((rep_num, T))
inv_series = np.zeros((rep_num, T))



for r in range(rep_num):
    
    # 从稳态出发
    x_path[r,0,:] = x_ss

    # 随机冲击
    eps = rng.normal(loc=0.0, scale=sigma_eps, size=T) 

    for t in range(T):
        x_t = x_path[r,t,:]

        # control
        y_t = (policy_mat @ x_t.reshape(-1,1)).reshape(-1)   
        y_path[r,t,:] = y_t

        # transition
        x_next = (trans_mat @ x_t.reshape(-1,1) + shock_mat * eps[t]).reshape(-1)
        x_path[r,t+1,:] = x_next

        # unpack levels
        k_t  = x_t[1]
        z_t  = x_t[2]
        kp_t = y_t[0]     
        h_t  = y_t[1]

        # implied variables (levels)
        y_out = np.exp(z_t) * (k_t**theta) * (h_t**(1-theta))
        c_t   = y_out + (1-delta)*k_t - kp_t
        inv_t = kp_t - (1-delta)*k_t

        k_series[r,t]   = k_t
        z_series[r,t]   = z_t
        kp_series[r,t]  = kp_t
        h_series[r,t]   = h_t
        y_series[r,t]   = y_out
        c_series[r,t]   = c_t
        inv_series[r,t] = inv_t






#%%----------------------------------------------------------------------
#--------------------HP filter--------------------
#----------------------------------------------------------------------

# 转稳态对数
y_hat   = 100.0*np.log(y_series  / y_ss)
c_hat   = 100.0*np.log(c_series  / c_ss)
inv_hat = 100.0*np.log(inv_series/ inv_ss)
k_hat   = 100.0*np.log(k_series  / k_ss)
h_hat   = 100.0*np.log(h_series  / h_ss)

data_hat = np.stack([y_hat, c_hat, inv_hat, k_hat, h_hat], axis=1) 

# hp
lamb = 1600
cycle = np.empty_like(data_hat)  
for r in range(rep_num):
    for j in range(5):
        cyc, tr = hpfilter(data_hat[r, j, :], lamb=lamb)
        cycle[r, j, :] = cyc



#%%----------------------------------------------------------------------
#--------------------计算标准差和相关系数--------------------
#----------------------------------------------------------------------
std_mat = np.std(cycle, axis=2, ddof=1)    

corr_mat = np.empty((rep_num, 5))
for r in range(rep_num):
    y_cyc = cycle[r, 0, :]
    for j in range(5):
        x_cyc = cycle[r, j, :]
        corr_mat[r, j] = np.corrcoef(y_cyc, x_cyc)[0, 1]

# 多次取平均
std_mean = std_mat.mean(axis=0)
std_sd   = std_mat.std(axis=0, ddof=1)

corr_mean = corr_mat.mean(axis=0)
corr_sd   = corr_mat.std(axis=0, ddof=1)



#%%----------------------------------------------------------------------
#--------------------输出结果--------------------
#----------------------------------------------------------------------
names = ["output", "consumption", "investment", "capital stock", "hours"]


print("\nEconomy with divisible labor")
print("                     \t     std(x) \t corr(y,x)")
for i in range(5):
    print(f"{names[i]:<20s}\t {std_mean[i]:5.2f} ({std_sd[i]:4.2f})\t {corr_mean[i]:5.2f} ({corr_sd[i]:4.2f})")