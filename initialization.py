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
    This class represents the algorithm to smartly initialize the gaussian dictionnary learning
    algorithm.
    To use it, create an instance, pass the appropriate parameters, and fit it.

    After fitting the input data, the results can be accessed thorugh the attributes
    of the algorithm object you created:
        - vertices: the spectra found
        - coefficients: the activation matrix
    """
    
    def __init__(self, data, num_phases):
        """    
        Create an instance of the algorithm
        :param data: The array of pixels
        :param num_phases: The number of phases to extract from the data
        """
        #Data related variables
        self.nb_pixel = data.shape[0]
        self.data = data
        self.num_phases = num_phases
    
        #Intermediate variables
        self.proj_rd = zeros((self.nb_pixel, num_phases -1)) #On principal components (Reduction Dimension)
        self.proj_vertices = zeros((num_phases, num_phases -1)) #Projected vertices
        self.proj_ch = zeros((self.nb_pixel, num_phases -1)) #On Convex Hull
        self.base = zeros((num_phases, num_phases - 1)) #Orthonormal base of the first projection space
        
        #Result variables 
        self.vertices = zeros((num_phases, data.shape[1]) )
        self.coefficients = zeros((self.nb_pixel, num_phases))

    def _find_min_point(self, P):
        """
        Finding the minimum point in the convex hull of a finite set of points.
        Based on the work of Philip Wolf and the recursive algorithm of Kazuyuki Sekitani and Yoshitsugu Yamamoto.
        https://scipy-cookbook.readthedocs.io/items/Finding_Convex_Hull_Minimum_Point.html
        :param P: the vertices of the polytope to project on
        :result: the projection of the origin on it
        """

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
        """
        The Gram-Schmidt orthonormalisation processus applied to 'base'
        :param base: an array of vectors:
        :result: an orthonormal base 'b' with Span(b) = Span('base')
        """

        for i in range(len(base)):
            for j in range(i):
                base[i] -= dot(base[i],base[j])*base[j]
            base[i] = base[i]/norm(base[i])

        return base

    def _projection(self, point,base):
        """
        Project orthogonnaly 'point' on the subspace generated by 'base'
        :param point: a vector
        :param base: an array of vectors
        :result: the orthogonal projection of 'point' onto Span('base')
        """

        projected = zeros(point.shape)
        for b in base:
            projected += dot(point,b)*b

        return projected

    def _distance(self, point, base):
        """
        Compute the distance between 'point' and the subspace generated by 'base'
        :param point: a vector
        :param base: an array of vectors
        :result: the distance between 'point' and Span('base')
        """

        point2 = point - base[0]
        base2 = base - base[0]
        base2 = base2[1:]

        base2 = self._orthonormalisation(base2)
        projected = self._projection(point2,base2)

        return norm(point2 - projected)

    def _compute_vertices(self):
        """
        Compute the vertices of the convex hull of proj_rd
        """

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

        self.proj_vertices = self.proj_rd[vector_indices]
        self.vertices = self.data[vector_indices]
        self.base = base

    def fit(self):
        """
        Reduces the dimension
        Compute the vertices
        Project on the convex hull of the vertices
        Compute the coefficient
        """
        
        #Dimension reduction to filter noise
        pca = PCA(n_components=self.num_phases - 1)
        self.proj_rd = pca.fit_transform(self.data)
        
        #Compute vertices
        self._compute_vertices()
        
        #Project on convex hull
        for i in range(self.nb_pixel):
            translated_vertices = array([self.proj_vertices[k] - self.proj_rd[i] for k in range(self.num_phases)])
            self.proj_ch[i] = self._find_min_point(translated_vertices) + self.proj_rd[i]
        
        #Compute activation coeficients
        translated_data = self.proj_ch - self.base[0]
        new_base = array([self.base[k] - self.base[0] for k in range(1,self.num_phases)])
        inverse = inv(new_base)

        util_mat = identity(self.num_phases)[:-1]  #tools to compute the coefficients
        util_vect = zeros(self.num_phases)
        util_vect[-1] = -1
        util_mat += util_vect
        util_vect[-1] = 1

        self.coefficients = multi_dot([translated_data,inverse,util_mat]) + util_vect

if __name__=="__main__":

    X = load("Data/data.npy")
    picture_dim = X.shape[:2]
    X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
    init = Initialisation(X, 3)
    init.fit()

    Z = init.coefficients
    Y = init.proj_ch
    X = init.proj_rd
    
    if(init.num_phases == 3):
        plt.figure()
        plt.scatter(init.proj_rd[:,0],init.proj_rd[:,1], color='green', s=1) 
        #plt.scatter(init.proj_ch[:,0],init.proj_ch[:,1], color='blue')   #Shows the projection on the polytope
        plt.scatter(array([init.base[0],init.base[1],init.base[2]])[:,0],array([init.base[0],init.base[1],init.base[2]])[:,1], color='red')
        plt.show()
    elif(init.num_phases == 4):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(init.proj_rd[:,0],init.proj_rd[:,1], init.proj_rd[:,2], color='green')
        #ax.scatter(init.proj_ch[:,0],init.proj_ch[:,1], init.proj_rd[:,2], color='blue')   #Shows the projection on the polytope
        ax.scatter(init.base[:,0],init.base[:,1], init.base[:,2], color='red', s = 200)
        plt.show()
    
    fig2 = plt.figure(figsize=(5.5*init.num_phases,4.5))
    for i in range(init.num_phases):
        plt.subplot(1,init.num_phases,i+1)
        plt.imshow(init.coefficients.reshape(*picture_dim, init.num_phases)[:, :, i], cmap="viridis")
        plt.grid(b=30)
        plt.title("Activations of first spectrum")
        plt.colorbar()
        plt.clim(0, 1)
    plt.show()
