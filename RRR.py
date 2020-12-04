import numpy as np

class RRR:
    def __init__(self, d, mode, cov_reg = 0):
        self.d = d
        self.mode = mode
        self.cov_reg = cov_reg
        
    def fit(self,X,Y):
        
        samples, X_dim = X.shape; Y_dim = Y.shape[1]
        assert Y.shape[0] == X.shape[0], 'Number of X and Y samples must be the same.' 

        Cxx = X.T@X/samples + self.cov_reg*np.eye(X_dim)
        Cyy = Y.T@Y/samples + self.cov_reg*np.eye(Y_dim)

        xsvd = np.linalg.svd(Cxx)
        ysvd = np.linalg.svd(Cyy)


        X_resc = X@xsvd[0]@np.sqrt(np.diag(1/xsvd[1]))@xsvd[2]


        if self.mode.lower() == 'cca':

            Y_resc = Y@ysvd[0]@np.sqrt(np.diag(1/ysvd[1]))@ysvd[2]
            cross_svd = np.linalg.svd(X_resc.T@Y_resc/samples)

            X_orthog_w = cross_svd[0].T
            Y_orthog_w = cross_svd[2]

            x_weights = X_orthog_w@xsvd[0]@np.diag(xsvd[1]**-0.5)@xsvd[0].T
            y_weights = (np.linalg.inv(Cyy)@Y.T@X@x_weights.T/samples).T;

        elif self.mode.lower() == 'rrmse':

            Y_resc = Y
            cross_svd = np.linalg.svd(X_resc.T@Y_resc/samples)

            X_orthog_w = cross_svd[0].T
            Y_orthog_w = cross_svd[2]

            x_weights = X_orthog_w@xsvd[0]@np.diag(xsvd[1]**-0.5)@xsvd[0].T
            y_weights = (Y.T@X@x_weights.T/samples).T;

        else:
            raise ValueError('Mode can only be CCA and RRMSE.')
            
        
        self.x_weights = x_weights[:self.d]
        self.y_weights = y_weights[:self.d]
        
        
    def compute_loss(self,X,Y):
        
        Sigma_inv = Y.T@Y/Y.shape[0] if self.mode=='cca' else np.eye(Y.shape[1])
        
        diff = Y - X@self.x_weights.T@self.y_weights@Sigma_inv
        
        return (np.sum((diff@np.linalg.inv(Sigma_inv))*diff)/Y.shape[0])