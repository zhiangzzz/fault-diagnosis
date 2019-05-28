import numpy as np
import matplotlib.pyplot as plt


class SVDD:
    def __init__(self, C = 0.1,kTup = ['gaussian',100,1,2]):
        self.C = C
        self.kTup = kTup

    def read_txt(self,x,y):
        self.x_train = np.loadtxt(x).T
        self.x_test = np.loadtxt(y)
        return self.x_train,self.x_test

    def yuchuli(self,x_train,x_test):
        m1 = x_train.shape[0]
        m2 = x_test.shape[0]
        mean_x = np.tile(np.mean(x_train,axis=0),(m1,1))
        std_x = np.tile(np.std(x_train,axis=0),(m1,1))
        x_train1 = (x_train-mean_x)/std_x
        mean_x2 = np.tile(np.mean(x_train, axis=0), (m2, 1))
        std_x2 = np.tile(np.std(x_train, axis=0), (m2, 1))
        x_test1 = (x_test-mean_x2)/std_x2
        return x_train1,x_test1

    def Kernel_matrix(self,X1,X2):
        m1 = X1.shape[0]
        m2 = X2.shape[0]
        K = np.zeros([m1,m2])
        if self.kTup[0] == 'gaussian' :
            for i in range(m1):
                for j in range(m2):
                    tmp = np.linalg.norm(X1[i,:]-X2[j,:])
                    K[i,j] = np.exp(-(tmp**2)/(self.kTup[1]**2))
        elif self.kTup[0] == 'linear' :
            K = np.dot(X1,X2.T)
        elif self.kTup[0] == 'sigmiod':
            g = self.kTup[1]
            c = self.kTup[2]
            K = np.tanh(g*np.dot(X1,X2.T)+c)
        elif self.kTup[0] == 'poly':
            g = self.kTup[1]
            c = self.kTup[2]
            n = self.kTup[3]
            K = (g*np.dot(X1,X2.T)+c)**n
        return K

    def SVDD_smo(self,K):
        L = np.shape(K)[0]
        alpha = np.zeros((1,L))
        r = np.random.random_integers(0,L,(1,1))
        r = int(r)
        alpha[0][r] = 1
        G = np.zeros((1,L))

        for i in range(L):
            if alpha[0][r] > 0:
                G[:] = G[:] + alpha[0][i]*K[:,i]
        while True:
            i,j,b_exit = self.SVDD_select_sets(alpha,G,L)
            if b_exit:
                break
            else:
                old_alpha_i = alpha[0][i]
                old_alpha_j = alpha[0][j]

                delta = (G[0][i]-G[0][j])/max(K[i,i] + K[j,j] -2*K[i,j],0)
                sum = alpha[0][i] + alpha[0][j]

                alpha[0][j] = alpha[0][j] +delta
                alpha[0][i] = alpha[0][i] - delta

                if alpha[0][i] < 0:
                    alpha[0][i] = 0
                    alpha[0][j] = sum

                if alpha[0][j] < 0 :
                    alpha[0][j] = 0
                    alpha[0][i] = sum

                if alpha[0][i] > self.C:
                    alpha[0][i] = self.C
                    alpha[0][j] = sum - self.C

                if alpha[0][j] > self.C:
                    alpha[0][j] = self.C
                    alpha[0][i] = sum - self.C

                delta_alpha_i = alpha[0][i] - old_alpha_i
                delta_alpha_j = alpha[0][j] - old_alpha_j

                G[:] = G[:] + K[:,i]*delta_alpha_i + K[:,j]*delta_alpha_j
        return alpha




    def SVDD_select_sets(self,alpha,G,L):
        Gmax1 = -float('inf')
        Gmax1_idx = -1
        Gmax2 = -float('inf')
        Gmax2_idx = -1

        eps = 1e-5
        for i in range(L):
            if alpha[0][i] < self.C:
                if -(G[0][i]) > Gmax1+(1e-15) :
                    Gmax1 = -(G[0][i])
                    Gmax1_idx = i

            if alpha[0][i] > 0 :
                if G[0][i] > Gmax2 +(1e-15) :
                    Gmax2 = G[0][i]
                    Gmax2_idx = i

        s = Gmax1_idx
        t = Gmax2_idx

        if Gmax1 + Gmax2 < 0.5*eps:
            b_exit = 1
        else:
            b_exit = 0

        return s,t,b_exit


    def CaculateR2D2(self,x_train,x_test,alpha,K):
        m,n = x_train.shape
        m2 = x_test.shape[0]
        sv_index = []
        for i in range(m):
            if alpha[0][i] > 0:
                sv_index.append(i)
        sv = np.zeros((len(sv_index), n))
        for i in range(len(sv_index)):
            for i1 in range(n):
                sv[i][i1] = x_train[i][i1]
        K2 = self.Kernel_matrix(x_test,x_train)
        K3 = self.Kernel_matrix(sv,x_train)

        D2 = 1 + np.sum(np.sum(np.dot(alpha.T,alpha)*K,axis=1))-2*np.sum(np.tile(alpha,[m2,1])*K2,axis=1)
        #D2 = np.sqrt(D2)
        R2 = 1 + np.sum(np.sum(np.dot(alpha.T,alpha)*K,axis=1))-2*np.sum(np.tile(alpha,[len(sv_index),1])*K3,axis=1)
        R2 = np.sqrt(max(R2))
        R2 = np.ones((m2,1))*R2
        return R2,D2

    def Figzz(self,R2,D2):
        plt.figure(1)
        plt.plot(D2)
        plt.plot(R2)
        plt.show()
        print('ok')







if __name__=='__main__':
    x = '/home/rapperli/practice_project/SVDD_faultdiagnosis/xunlian/train0.txt'
    y = '/home/rapperli/practice_project/SVDD_faultdiagnosis/ceshi/test1.txt'
    XX = SVDD(0.1,['gaussian',100,2,3])

    x_train, x_test= XX.read_txt(x,y)
    x_train, x_test = XX.yuchuli(x_train, x_test)
    K = XX.Kernel_matrix(x_train,x_train)
    alpha = XX.SVDD_smo(K)
    R2,D2 = XX.CaculateR2D2(x_train,x_test,alpha,K)
    XX.Figzz(R2,D2)
    A = D2.T-R2
    zz = 1
    lis = []
    for i in range(np.size(A,1)):
        aaa = A[0,i]
        if aaa < 0:
            lis.append(zz)
        zz += 1
    print(lis)

