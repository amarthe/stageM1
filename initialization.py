from random import randint  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array, matrix, sin, sqrt, dot, cos, ix_, zeros, concatenate, abs, log10, exp, ones, load, zeros, identity
from numpy.linalg import norm, inv, multi_dot

from mpmath import mpf, mp
mp.dps=80

class Initialisation:
    """
    TODO
    """
    
    def __init__(self, data, num_phases):
        
        #Data related variables
        self.nb_pixel = X.shape[0] * X.shape[1]
        self.picture_dim = X.shape[:2]
        self.data = data.reshape(self.nb_pixel, X.shape[2])  #NOTE : maybe not necessary to reshape here
        self.num_phases = num_phases
    
        #Intermediate variables
        self.proj_rd = zeros((self.nb_pixel, num_phases -1)) #On Principal components
        self.proj_ch = zeros((self.nb_pixel, num_phases -1)) #On Convex Hull
        self.base = zeros((num_phases, num_phases - 1)) #Orthonormal base of the first projection space
        
        #Result variables 
        self.vertices = zeros((num_phases, X.shape[1]) )
        self.coefficients = zeros((self.nb_pixel, num_phases))

    def _find_min_point(self, P):

        if len(P) == 1:
            return P[0]

        eps = mpf(10)**-10 #NOTE : precision value can be changed

        P = [array([mpf(i) for i in p]) for p in P]

        # Step 0. Choose a point from C(P)
        x  = P[array([dot(p,p) for p in P]).argmin()]

        while True:

            # Step 1. \alpha_k := min{x_{k-1}^T p | p \in P}
            p_alpha = P[array([dot(x,p) for p in P]).argmin()]
            if dot(x,x-p_alpha) < eps:
                return array([float(i) for i in x])
            Pk = [p for p in P if abs(dot(x,p-p_alpha)) < eps]

            # Step 2. P_k := { p | p \in P and x_{k-1}^T p = \alpha_k}
            P_Pk = [p for p in P if not array([(p == q).all() for q in Pk]).any()]
            if len(Pk) == len(P):
                return array([float(i) for i in x])
            y = self._find_min_point(Pk)
            p_beta = P_Pk[array([dot(y,p) for p in P_Pk]).argmin()]
            if dot(y,y-p_beta) < eps:
                return array([float(i) for i in y])

            # Step 4. 
            P_aux = [p for p in P_Pk if (dot(y-x,y-p)>eps) and (dot(x,y-p)!=0)]
            p_lambda = P_aux[array([dot(y,y-p)/dot(x,y-p) for p in P_aux]).argmin()]
            lam = dot(x,p_lambda-y) / dot(y-x,y-p_lambda)
            x += lam * (y-x)

    def _orthonormalisation(self, base):

        for i in range(len(base)):
            for j in range(i):
                base[i] -= dot(base[i],base[j])*base[j]
            base[i] = base[i]/norm(base[i])

        return base

    def _projection(self, point,base):

        projected = zeros(point.shape)
        for b in base:
            projected += dot(point,b)*b
        return projected

    def _distance(self, point, base):

        point2 = point - base[0]
        base2 = base - base[0]
        base2 = base2[1:]

        base2 = self._orthonormalisation(base2)
        projected = self._projection(point2,base2)

        return norm(point2 - projected)

    def _compute_vertices(self):

        vector_indices = zeros(self.num_phases, dtype=int)

        r = randint(0,self.nb_pixel -1)
        v0 = self.proj_rd[r]

        index = array([norm(vect - v0) for vect in self.proj_rd]).argmax()
        vector_indices[0] = index
        base = array([self.proj_rd[index]])

        for k in range(1,self.num_phases):
            index = array([self._distance(vect, base) for vect in self.proj_rd]).argmax()
            vector_indices[k] = index
            base = array([*base,self.proj_rd[index]])

        self.vertices = self.proj_rd[vector_indices]
        self.base = base

    def fit(self):
        
        #Dimension reduction to filter noise
        pca = PCA(n_components=self.num_phases - 1)
        self.proj_rd = pca.fit_transform(self.data)
        
        #Compute vertices
        self._compute_vertices()
        
        #Project on convex hull
        for i in range(self.nb_pixel):
            translated_vertices = array([self.vertices[k] - self.proj_rd[i] for k in range(self.num_phases)])
            self.proj_ch[i] = self._find_min_point(translated_vertices) + self.proj_rd[i]
        
        #Compute activation coefficients
        translated_data = self.proj_ch - self.base[0]
        new_base = array([self.base[k] - self.base[0] for k in range(1,self.num_phases)])
        inverse_mat = inv(new_base)
        
        #TODO : change to simplify and reduce complexity a bit
        util_mat = identity(self.num_phases)[:-1]
        util_vect = zeros(self.num_phases)
        util_vect[-1] = -1
        util_mat += util_vect
        util_vect[-1] = 1

        self.coefficients = multi_dot([translated_data,inverse_mat,util_mat]) + util_vect


if __name__=="__main__":

    X = load("Data/data_gaussian.npy")
    init = Initialisation(X, 3)
    init.fit()
    
    ##Plot 2D
    plt.figure()
    plt.scatter(init.proj_rd[:,0],init.proj_rd[:,1], color='green')
    plt.scatter(init.proj_ch[:,0],init.proj_ch[:,1], color='blue')
    plt.scatter(array([init.base[0],init.base[1],init.base[2]])[:,0],array([init.base[0],init.base[1],init.base[2]])[:,1], color='red')
    plt.show()
    
    ##Plot3D
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(init.proj_rd[:,0],init.proj_rd[:,1], init.proj_rd[:,2], color='green')
    #ax.scatter(init.proj_ch[:,0],init.proj_ch[:,1], init.proj_rd[:,2], color='blue')
    #ax.scatter(init.base[:,0],init.base[:,1], init.base[:,2], color='red', s = 20)
    #plt.show()
    
    fig2 = plt.figure(figsize=(5.5*init.num_phases,4.5))
    for i in range(init.num_phases):
        plt.subplot(1,init.num_phases,i+1)
        plt.imshow(init.coefficients.reshape(*init.picture_dim, init.num_phases)[:, :, i], cmap="viridis")
        plt.grid(b=30)
        plt.title("Activations of " + str(i+1) + "e  spectrum")
        plt.colorbar()
        plt.clim(0, 1)
    plt.show()
