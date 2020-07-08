
import numpy as np
import pandas as pd
from scipy import stats

#_____________ Gerenal _____________

def border(pattern:str, length:int): # 框線
    print(pattern*length)

def var_list(v, l): #自動產生變數名稱
    if v == None: 
       return ["v"+str(i+1) for i in range(l)]
    else:
        return v

#_____________ Description _____________


#_____________ Variance _____________

def cov(x, y=None, df=0):
    '''
        Covariance matrix  
        cp = Σ(x-xbar)*(y-ybar), 離均差交乘積和  
        cxy = cp/n, 共變異數  
    '''
    return np.cov(x, y, ddof=df) # 離均差平方和之平均

#_____________ Hypothesis _____________
#1.常態分配 
#2.變異數相等
#3.期望值落在一直線上
#4.Y之間互相獨立

class Linear:
    def __init__(self, x, y=None, varName:list=None):
        '''if y is None, the last row of x matrix defaults to y.'''
        self.x = np.array(x)
        self.y = np.array(y)
        self.name = varName
        if y == None:
            self.y = np.array(self.x[ -1, :]) # np slice
            self.x = np.array(self.x[:-1, :])
        x_1 = np.r_[[[1]*len(self.y)], self.x] # x for covar, x_1 for matrix [1, x1, ...]
        self.n = len(self.y) # subj. number
        self.k = len(x_1)    # para. number
        if varName == None:
            self.name = ["var"+str(i+1) for i in range(len(self.x))]
            self.name.append('y')
        self.xTx = np.dot(   x_1, np.transpose(x_1)) # 順序?
        self.xTy = np.dot(self.y, np.transpose(x_1))
        self.ssT, self.ssR, self.ssE = 0, 0, 0 # ss()
        self.msR, self.msE = 0, 0 # anova

    def mtx_B(self):
        '''Matrices approach to β'''
        return np.dot(self.xTy, np.linalg.inv(self.xTx))

    def corr(self):
        '''
            Correlation coefficients  
            r = σxy / (σx*σy)^(.5)  
              = np.cov(y, x) / (np.std(x)*np.std(y))  
              = (SSR / SST)^(.5)
            以樣本r推論母群ρ時, 標準差分母 n-1, 相關係數略小  
            --規約--  
            0.7 <= r < 1,   高  
            0.3 <= r < 0.7, 中  
            0   <  r < 0.3, 低  
        '''
        return np.corrcoef(self.x, self.y)

    def corr_tb(self):
        r = self.corr()
        t = np.zeros(shape=np.shape(r))
        p = np.zeros(shape=np.shape(r))
        # show correlation table
        print("\n【Correlations】 (n={:d})".format(self.n))
        border("-", 47)
        print("{:8}\t{}\t{}\t{}\t{}".format("Pearson","r","t","df","p"))
        border("-", 47)
        for i in range(np.shape(r)[0]):
            for j in range(np.shape(r)[1]):
                if i < j: # 優化, 上三角形
                    mse = ((1-r[i][j]**2)/(self.n-2))**.5
                    if mse == 0:
                        print("{:10}--{}\t{}".format(self.name[i],self.name[j],"完全重疊"))
                    else:
                        t[i][j] = r[i][j] / mse  # 分母為標準誤
                        p[i][j] = 1 - stats.t.cdf(abs(t[i][j]), df=self.n-2) # 還沒設定單雙尾、自由度, 計算cdf時要小心t值可能是負的
                        print("{:4}--{:4}\t{:.3f}\t{:.2f}\t{:d}\t{:.3f}".format(self.name[i],self.name[j],r[i][j],t[i][j],self.n-2,p[i][j]))
        border("=", 47)

    def ss(self):
        '''
            SSR = b'X'Y - (1/n)*Y'JY
            SSE = Y'Y - b'X'Y
            SST = Y'Y - (1/n)*Y'JY

            乘積級距公式 Product-moment formula  
            y = ax +b,   x = cy +d  
            a = σxy/σ2x, c = σxy/σ2y
            r = ( bxy * byx )**.5

            # 標準方程組
            1. Σy   = b0*n   + b1*Σx1    + b2*Σx2
            2. Σx1y = b0*Σx1 + b1*Σx1**2 + b2*Σx1x2
            3. Σx2y = b0*Σx2 + b1*Σx1x2  + b2*Σx2**2

            ## b == (x'x)^(-1)*(x'y)
            
            # 矩陣  
            [bo]   [  n  Σx1   Σx2  ]   [Σy  ]  
            [b1] * [ Σx1 Σx1^2 Σx1x2] = [Σx1y]  
            [b2]   [ Σx2 Σx1x2 Σx2^2]   [Σx2y]  
        '''
        b = self.mtx_B()
        J = np.ones((self.n, self.n))
        yTy = np.dot(self.y, np.transpose(self.y))
        yTJy = self.y.dot(J).dot(np.transpose(self.y))
        self.ssT = yTy - (1/self.n)*yTJy
        self.ssR = b.dot(self.xTy) - (1/self.n)*yTJy # b為1D怎麼擺都可以
        #self.ssE = yTy - b.dot(self.xTy)
        self.ssE = self.ssT - self.ssR

    def anova_tb(self):
        if self.ssE == 0:
            self.ss()
        dfR, dfE = self.k-1, self.n-self.k
        self.msR, self.msE = self.ssR/dfR, self.ssE/dfE
        f = self.msR/self.msE
        print("\n【ANOVA】")
        border("-", 72)
        print("{:^8}\t{}\t{:>3}\t{}\t{}\t{:^}".format("Model","Sum of Squares", "df", "Mean Square", "F", "Sig."))
        border("-", 72)
        print("{:8}\t{:8.2f}\t{:3.0f}\t{:8.2f}\t{:.2f}\t{:.3f}".format("Regression",self.ssR, dfR, self.msR, f, 1-stats.f.cdf(f,dfR,dfE)))
        print("{:8}\t{:8.2f}\t{:3.0f}\t{:8.2f}                ".format("Residual",  self.ssE, dfE, self.msE))
        print("{:8}\t{:8.2f}\t{:3.0f}                         ".format("Total",     self.ssT, self.n-1))
        border("=", 72)

    def beta(self, b):
        '''
            矩陣法?
        '''
        beta = [0.0]*(len(b)-1)
        for i in  range(len(b)-1):
            beta[i] = b[i+1] * pow(cov(self.x[i])/cov(self.y), .5) # const have no beta
        return [""] + beta

    def vif(self):
        '''
            VIF = 1/ (1-Rj2)  
            xj作為dv, 與其他x的判定係數Rj2  
            建議VIF > 10, 該變數需刪除  
            觀察：  
            1. 整體 F、R2很大，但個別判定係數 t 很小  
            補救：
            1.刪除x
            2.增加n
            3.用 脊迴估計量
            4.用 PCA變數 取代共線性變數
        '''
        return

    def coef_tb(self):
        '''
            β1檢定  
            Var(β1_hat) = σ**2 / Σ(Xi-Xmean)**2  
            E(MSE) = σ**2  
            E(MSR) = σ**2 + β1**2 * Σ(Xi-Xmean)**2  
            T = (β1 - b1) / ( MSE * (1/ Σ(Xi-Xmean)**2)**.5 ) > tα(n-2)  

            β0檢定  
            Var(β0_hat) = [1/n + Xmean**2 / Σ(Xi-Xmean)**2] * σ**2  
            T = (β0 - b0) / ( MSE * (Xmean**2 / (1/n +  Σ(Xi-Xmean)**2))**.5 )  

            b = Sxy/Sxx = rxy * Sy/Sx  
            beta = Sxy/(Sxx*Syy)**.5  
        '''
        r = self.corr()
        b = self.mtx_B()
        beta = self.beta(b)
        if self.msE == 0:
            self.anova_tb()
        sb = self.msE*np.linalg.inv(self.xTx) # 負數導致 Nan, 找對角線
        print("\n【Coefficients】")
        border("-", 118)
        print("|     | Unstandardized  |Standardized|                |_95%_CI_for_B__|                           |                  |")
        print("|_____|__Coefficients___|Coefficients|________________| Lower | Upper |_______Correlations________|___Collinearity___|")
        print("|Model|___B___|Std.Error|_____Beta___|____t_____Sig.__|_Bound_|_Bound_|__Zero__|_Partial_|__Part__|_Tolerance_|__VIF_|")
        border("-", 118)
        print("{}"     .format("const")              ,end='\t')
        print("{:4.3f}".format(b[0])                 ,end='\t')
        print("{:7.3f}".format(pow(sb[0][0],.5))     ,end='\t')
        print("{:>8}"  .format("")                   ,end='\t')
        print("{:.3f}" .format(b[0]/pow(sb[0][0],.5)),end='\t')
        print("{:.3f}" .format(1-stats.t.cdf(abs(b[0]/pow(sb[0][0],.5)),df=self.n-self.k)),end='\n')
        for s, t in enumerate(self.name[:-1]): 
            s += 1
            t_value = b[s]/pow(sb[s][s],.5)
            print("{}"      .format(t)                ,end='\t')
            print("{:4.3f}" .format(b[s])             ,end='\t')
            print("{:7.3f}" .format(pow(sb[s][s],.5)) ,end='\t')
            print("{:11.3f}".format(beta[s])          ,end='\t')
            print("{:.3f}"  .format(t_value)          ,end='\t')
            print("{:.3f}"  .format(2*(1-stats.t.cdf(abs(t_value),df=self.n-self.k))),end='\t')
            print("{}"      .format("LB")             ,end='\t')
            print("{}"      .format("UB")             ,end='\t')
            print("{:.3f}"  .format(r[s][-1])         ,end='\n')
        border("=", 118)
        print("DV：{}、Sig.t為雙尾檢定".format(self.name[-1]))

    def report(self):
        self.corr_tb()
        self.anova_tb()
        self.coef_tb()
        return ""

    def model(self):
        return

kk = Linear(x,y).report()
print(kk)

#_____________ Correlation _____________

'''Fisher檢定
當 ρ ≠ 0時
Z = (Zr - Zρ) /( 1/ (n-3)**.5)
'''

def corr_rank(x, y):
    '''
        Spearman's rank correlation coefficient
    '''
    a = x # 轉為rank
    b = y
    n = len(x)
    d = [(a[i] - b[i])**2 for i in range(n)]
    r_rank = 1 - 6 * sum(d) / (n**3-n)
    # 小樣本時 t 同pearson
    # 大樣本時 z = r * (n-1)**.5
    return r_rank


#_____________ Logistic _____________

class Logistic():
    def __init__(self):
        return

#_____________ Method of Moment, MOM _____________

#_____________ Ordinary Least Square Estimate, OLSE _____________

def OLSE():
    '''
        最小平方法, 定理:
        1. 配方法求極值
        2. 柯西−施瓦茲不等式
        3. 相關係數

        Gauss-Markov Theorem  
        當 誤差 滿足 1.零均值 2.同標準差 3.彼此獨立  
        則 迴歸係數的最佳線性無偏估計(BLUE)就是LS  
    '''
    return

def aic(): 
    '''
    Akaike's Information Criterion (赤池訊息量準則,AIC)
    AIC = 2k - 2ln(L) # k為參數數量、L是似然函數
    AIC = 2k + n*ln(RSS/n) # n為觀察數、RSS為殘差平方和
    小樣本時,
        AICc = AIC + 2k(k+1)/(n-k-1)
    '''
    return

def bic():
    '''
    Bayesian Information Criteria (BIC)
    BIC = ln(n)*k - 2ln(L) # 同AIC, 越小表示模式的解釋力越好
    與AIC相比, 當n>=8時(多數情況的迴歸都是大於), 
        參數k越多, BIC產生的懲罰項就越大, 因此BIC更傾向於選擇參數少的簡單模型
    '''
    return

#_____________ Maximum Likelihood Estimate, MLE _____________

def ML():
    return

#_____________ Model Selection _____________

def cs():  # 柯西−施瓦茲不等式 Cauchy-Schwarz
    return

def Mallow_Cp():
    return

def hq():
    '''
        Hannan-Quinn Criterion (HQ)
        HQ = ln(ln(n))*k - 2ln(L)
    '''
    return




