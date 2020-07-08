
import numpy as np
import pandas as pd
from scipy import stats

#_____________ Gerenal _____________

def border(pattern:str, length:int): # 框線
    print(pattern*length)

#_____________ Description _____________

#_____________ Variance _____________

def cov(x, y=None, df=0):
    '''Covariance matrix'''
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
        self.xTx = np.dot(   x_1, np.transpose(x_1))
        self.xTy = np.dot(self.y, np.transpose(x_1))
        self.ssT, self.ssR, self.ssE = 0, 0, 0 # ss()
        self.msR, self.msE = 0, 0 # anova

    def mtx_B(self):
        '''Matrices approach to β'''
        return np.dot(self.xTy, np.linalg.inv(self.xTx))

    def corr(self):
        '''Correlation coefficients'''
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
                if i < j: # 上三角形
                    mse = ((1-r[i][j]**2)/(self.n-2))**.5
                    if mse == 0:
                        print("{:10}--{}\t{}".format(self.name[i],self.name[j],"完全重疊"))
                    else:
                        t[i][j] = r[i][j] / mse
                        p[i][j] = 1 - stats.t.cdf(abs(t[i][j]), df=self.n-2)
                        print("{:4}--{:4}\t{:.3f}\t{:.2f}\t{:d}\t{:.3f}".format(self.name[i],self.name[j],r[i][j],t[i][j],self.n-2,p[i][j]))
        border("=", 47)

    def ss(self):
        '''
            SSR = b'X'Y - (1/n)*Y'JY
            SSE = Y'Y - b'X'Y
            SST = Y'Y - (1/n)*Y'JY
        '''
        b = self.mtx_B()
        J = np.ones((self.n, self.n))
        yTy = np.dot(self.y, np.transpose(self.y))
        yTJy = self.y.dot(J).dot(np.transpose(self.y))
        self.ssT = yTy - (1/self.n)*yTJy
        self.ssR = b.dot(self.xTy) - (1/self.n)*yTJy
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
        beta = [0.0]*(len(b)-1)
        for i in  range(len(b)-1):
            beta[i] = b[i+1] * pow(cov(self.x[i])/cov(self.y), .5) # const have no beta
        return [""] + beta

    def coef_tb(self):
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
        '''Model Select'''
        return

    
