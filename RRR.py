import numpy as np

class RRR:
    def __init__(self, d, mode, cov_reg = 0):
        self.d = d
        self.mode = mode
        self.cov_reg = cov_reg
        
    def fit(self,X,Y):
        
        X_dim, samples = X.shape; Y_dim = Y.shape[0]
        assert Y.shape[1] == X.shape[1], 'Number of X and Y samples must be the same.' 
        
        Cxx = X@X.T/samples + self.cov_reg*np.eye(X_dim)
        Cyy = Y@Y.T/samples + self.cov_reg*np.eye(Y_dim)
        
        xsvd = np.linalg.svd(Cxx)
        ysvd = np.linalg.svd(Cyy)
        
        
        X_resc = xsvd[0]@np.sqrt(np.diag(1/xsvd[1]))@xsvd[2]@X
        
        if self.mode.lower() == 'cca':
            
            Y_resc = ysvd[0]@np.sqrt(np.diag(1/ysvd[1]))@ysvd[2]@Y
            cross_svd = np.linalg.svd(X_resc@Y_resc.T/samples)
            
            X_orthog_w = cross_svd[0].T
            Y_orthog_w = cross_svd[2]
            
            y_weights = Y_orthog_w@ysvd[0]@np.diag(ysvd[1]**-0.5)@ysvd[0].T
            x_weights = X_orthog_w@xsvd[0]@np.diag(xsvd[1]**-0.5)@xsvd[0].T
            
        elif self.mode.lower() == 'rrmse':
            
            Y_resc = Y
            cross_svd = np.linalg.svd(X_resc@Y_resc.T/samples)
            
            X_orthog_w = cross_svd[0].T
            Y_orthog_w = cross_svd[2]
            
            x_weights = X_orthog_w@xsvd[0]@np.diag(xsvd[1]**-0.5)@xsvd[0].T
            y_weights = (Y@X.T@x_weights.T/samples).T;
            
        else:
            raise ValueError('Mode can only be CCA and RRMSE.')
            
        self.x_weights = x_weights[:self.d]
        self.y_weights = y_weights[:self.d]
        self.coef_ = self.x_weights.T
        