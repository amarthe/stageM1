import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from scipy.sparse import lil_matrix

from utils import Gaussians, MatrixUtils


class DictionaryLearningAlgorithm:
    """
    This class represents the actual dictionary learning algorithm.
    To use it create an instance, and pass the appropriate parameters of the model.
    Then the fit method can be used to run the algorithm on your data. This method can
    be called repeatedly to further finetune, by setting the initialize argument to False.
    
    After fitting the input data, the results can be accessed through the attributes
    of the algorithm object you created:
        - p_matr: the num_phases columns of this matrix hold the estimated spectrum for every phase
                  represented as a weight by which the signals in the g_matr should be summed together
        - d_matr: this matrix is equal to the product of the g_matr with the d_matr, the columns hold
                  the estimated spectra for the phases in the energy domain
        - a_matr: the columns of this matrix hold the concentration values for every pixel: the concentration
                  of phase k, for pixel (i, j) is at a_matr[k, i*w+j] with w the width of the image
    """

    def __init__(self, num_phases, mu_lapl=0., mu_abs=0., mu_fro=0., g_matr=None, debug=True):
        """
        Create an instance of the dictionary learning algorithm
        :param num_phases: Number of phases to estimate
        :param mu_lapl: Strength of laplacian regularization,
        should be used to control spatial smoothness
        :param mu_abs: Strength of absolute value smoothness regularization,
        should be used to control spatial smoothness
        :param g_matr: (optional) Matrix with signals of which the learned spectra will be a linear combination of,
        if this is not specified the json at Data/xrays.json is used to compose these signals using the Gaussians
        class in utils.py, with the default arguments.
        :param debug: Enabling this flag will gather some metrics that can be used to debug the algorithm
        """
        # Set the parameters for the algorithm
        self.p_ = num_phases
        self.mu_lapl = mu_lapl
        self.mu_abs = mu_abs
        self.mu_fro = mu_fro
        self.debug = debug

        self.initialize_matrix = None
        self.first_column_fro = None

        # If no G matrix is provided, load the default one from Data/xrays.json
        if g_matr is None:
            self.g_matr = Gaussians().create_matrix()
        else:
            self.g_matr = g_matr

        # the activations matrix
        self.a_matr = None
        # the matrix containing the intensity of every signal in the G matrix, for every phase
        self.p_matr = None
        # dot product of G and P matrices, contains the actual learned spectra (cached for efficiency reasons)
        self.d_matr = None
        # the data matrix that we will try to fit
        self.x_matr = None
        # original shape of the data matrix (kept track of because X matrix is immediately reshaped when loaded)
        self.x_shape = None
        # dot product of the a matrix and the laplacian matrix (cached for efficiency reasons)
        self.a_dot_l = None
        # trace of the dot product X^T X (cached for efficiency reasons)
        self.tr_x_x = None

        # eta parameters that will be estimated with the line-search algorithm: step size is 1/eta
        self.eta_a = 1.
        self.eta_p = 1.

        # parameters for the line-search algorithm
        self.beta = 1.5
        self.alpha = 1.2

        # 'smoothness' of the absolute value used in the absolute value smoothness regularization
        self.abs_smoothness = 1e-8

        # to keep track of the number of iterations when fit is called multiple times
        self.num_iterations = None

        # holds the laplacian matrix of a, used for the laplacian smoothness regularization
        self._laplacian_a = None

        # for debugging purposes
        self.eta_as = []
        self.eta_ps = []
        self.base_losses = []

        # to observe the convergence of the algorithm
        self.losses = []
        self.p_update = []
        self.a_update = []
        self.a_norm = []
        self.p_norm = []

        # to keep the time of execution
        self.algo_time = 0.

    def _create_laplacian_a(self, n, m):
        """
        Helper method to create the laplacian matrix for the laplacian regularization
        :param n: number of pixels
        :param m: width of the original image
        :return:the n x n laplacian matrix
        """
        lap = lil_matrix((n, n), dtype=np.float32)
        lap.setdiag([2] + [3]*(m-1) + [4]*(n-2*m) + [3]*(m-1) + [2])
        lap.setdiag(-1, k=-1)
        lap.setdiag(-1, k=1)
        lap.setdiag(-1, k=m)
        lap.setdiag(-1, k=-m)
        return lap

    def _eval_function(self, a_matr=None, p_matr=None):
        """
        Evaluate the loss function that is being optimized:
        L(A, P) = .5 * ||X - GPA||^2 + .5 * mu_lapl * `sum of (diff in pixels)^2` + mu_abs * `sum of abs(diff in pixels) + .5 * mu_fro * (frobenius norme of GP[np.diag(0,1...1)])^2`
        :param a_matr: (optional) if you use this argument, the activations matrix
        stored in the class attributes is overriden
        :param p_matr: (optional) if you use this argument, the peaks matrix
        stored in the class attributes is overriden
        :return: the loss
        """
        if a_matr is None:
            a_matr = self.a_matr
            a_dot_l = self.a_dot_l
        else:
            if self.mu_lapl != 0.:
                a_dot_l = a_matr @ self._laplacian_a

        if p_matr is None:
            d_matr = self.d_matr
        else:
            d_matr = self.g_matr @ p_matr

        # for efficiency reasons: we don't need to calculate the matrix prod if mu_lapl = 0
        if self.mu_lapl != 0.:
            # calc the laplacian smoothness loss = .5 * mu_lapl * trace(ALA^T)
            spatial_loss_lapl = .5 * self.mu_lapl * np.einsum("ij,ij->", a_dot_l, a_matr)
        else:
            spatial_loss_lapl = 0.

        # for efficiency reasons: we don't need to calculate the matrix prod if mu_abs = 0
        if self.mu_abs != 0.:
            # calc the absolute value smoothness loss
            spatial_loss_abs = self.mu_abs * self._calc_spatial_abs_penalty()
        else:
            spatial_loss_abs = 0.

        # for efficiency reasons: we don't need to calculate the matrix prod if mu_fro = 0
        if self.mu_fro != 0.:
            #calc the Frobenius smoothness loss : .5 * mu_fro * ||GP||_2
            if self.first_column_fro:
                fro_loss = .5 * self.mu_fro * np.linalg.norm(np.delete(d_matr,0,1), 'fro')
            else:
                fro_loss = .5 * self.mu_fro * np.linalg.norm(d_matr, 'fro')
        else:
            fro_loss = 0.

        # add the three parts of the loss together
        return self._base_loss(a_matr, d_matr) + spatial_loss_abs + spatial_loss_lapl + fro_loss

    def _base_loss(self, a_matr=None, d_matr=None):
        """
        Evaluate the "base part" of the loss function, that is .5 * ||X - GPA||^2
        :param a_matr: (optional) if you use this argument, the activations matrix
        stored in the class attributes is overriden
        :param p_matr: (optional) if you use this argument, the peaks matrix
        stored in the class attributes is overriden
        :return: the base part of the loss
        """
        if a_matr is None:
            a_matr = self.a_matr
        if d_matr is None:
            d_matr = self.d_matr

        # calculate the base part of the loss
        # because tr_x_x = trace(XX^T) has to be calculated only once,
        # this computation is split in multiple parts for efficiency reasons:
        # ||X - GPA||^2 = trace(XX^T) + trace(X^TGPA) + trace((GP)^TGPAA^T)
        tr_x_d_a = np.einsum("ji,jk,ki->", self.x_matr, d_matr, a_matr, optimize="optimal")
        tr_dd_aa = np.einsum("ji,jk,kl,il->", d_matr, d_matr, a_matr, a_matr, optimize="optimal")

        return .5 * (self.tr_x_x - 2*tr_x_d_a + tr_dd_aa)

    def _calc_diff_matrices_a(self):
        """
        Calculates the matrices of differences in activations between neighbouring pixels, used to
        calculate the absolute value smoothness regularization
        :return: diff matrix with the pixel above, below, left, right, diagonally down, and diagonally up
        """
        a_matr = self.a_matr.T.reshape(self.x_shape[0], self.x_shape[1], self.p_).copy()
        a_down = MatrixUtils.shift_matrix_vert(a_matr, 1)
        a_up = MatrixUtils.shift_matrix_vert(a_matr, -1)
        a_left = MatrixUtils.shift_matrix_horiz(a_matr, 1)
        a_right = MatrixUtils.shift_matrix_horiz(a_matr, -1)
        a_diag_down = MatrixUtils.shift_matrix_diag(a_matr, 1)
        a_diag_up = MatrixUtils.shift_matrix_diag(a_matr, -1)
        return (a_matr-a_down,
                a_matr-a_up,
                a_matr-a_left,
                a_matr-a_right,
                a_matr-a_diag_down,
                a_matr-a_diag_up)

    def _calc_spatial_abs_penalty(self):

        """
        Calcultes the spatial smoothness through absolute values regularization loss
        :return: the spatial smoothness through absolute values regularization loss
        """
        a_down, a_up, a_left, a_right, a_diag_down, a_diag_up = self._calc_diff_matrices_a()

        return (self._smoothed_abs(a_down).sum() +
                self._smoothed_abs(a_up).sum() +
                self._smoothed_abs(a_left).sum() +
                self._smoothed_abs(a_right).sum() +
                self._smoothed_abs(a_diag_down).sum() / np.sqrt(2) +  # divide by sqrt(2) to account for diagonal dist
                self._smoothed_abs(a_diag_up).sum() / np.sqrt(2))

    def _calc_grad_spatial_abs_penalty(self):
        """
        Calcultes the gradient of the spatial smoothness through absolute values regularization loss
        with respect to the activations matrix
        :return: the spatial smoothness through absolute values regularization loss
        """
        a_down, a_up, a_left, a_right, a_diag_down, a_diag_up = self._calc_diff_matrices_a()

        return 2 * (self._deriv_smoothed_abs(a_down) +
                    self._deriv_smoothed_abs(a_up) +
                    self._deriv_smoothed_abs(a_left) +
                    self._deriv_smoothed_abs(a_right) +
                    self._deriv_smoothed_abs(a_diag_down / np.sqrt(2)) +
                    self._deriv_smoothed_abs(a_diag_up / np.sqrt(2)))\
            .reshape(self.x_shape[0]*self.x_shape[1], self.p_).T

    def _smoothed_abs(self, x):
        """
        The smoothed absolute value function sqrt(x^2 + eps) in which epsilon is chosen small (and is set as a class
        attribute abs_smoothness)
        :param x: input data
        :return: smoothed absolute value of input data
        """
        return np.sqrt(x**2 + self.abs_smoothness)

    def _deriv_smoothed_abs(self, x):
        """
        Derivative of the smoothed absolute value function sqrt(x^2 + eps)
        :param x: input data
        :return: derivative of the smoothed absolute value function, evaluated in x
        """
        return x / np.sqrt(x**2 + self.abs_smoothness)

    def _project_on_simplex(self, data):
        """
        Project the input data on the convex set in which the value
        of all columns are in [0, 1] and all columns sum to 1.
        Based on the algorithm described in https://arxiv.org/pdf/1309.1541.pdf
        :param data: data to project
        :return: data projected on simplex
        """
        if data.sum() == data.shape[1] and np.alltrue(data >= 0):
            # best projection: itself!
            return data

        # get the array of cumulative sums of a sorted (decreasing) copy of v
        u = np.sort(data, axis=0)[::-1]
        cssv = np.cumsum(u, axis=0)
        # get the number of > 0 components of the optimal solution
        rho = - np.argmax((u * np.arange(1, data.shape[0] + 1)[:, np.newaxis] > (cssv - 1))[::-1, :], axis=0) + \
              data.shape[0] - 1
        # compute the Lagrange multiplier associated to the simplex constraint
        theta = (cssv[rho, np.arange(len(rho))] - 1) / (rho + 1.0)
        # compute the projection by thresholding v using theta
        w = (data - theta).clip(min=0)

        return w

    def _gradient_a(self):
        """
        Calculate the gradient of the loss function with respect to a
        :return: gradient
        """
        # no need to to the expensive calculations if mu_lapl is 0
        if self.mu_lapl != 0.:
            # gradient of the laplacian part = mu_lapl * AL
            lapl_part = self.mu_lapl * self.a_dot_l
        else:
            lapl_part = 0.

        # no need to to the expensive calculations if mu_abs is 0
        if self.mu_abs != 0.:
            abs_part = self.mu_abs * self._calc_grad_spatial_abs_penalty()
        else:
            abs_part = 0.

        # gradient of the base part = -P^TG^T(X + GPA)
        return - self.d_matr.T @ self.x_matr + np.linalg.multi_dot((self.d_matr.T, self.d_matr, self.a_matr)) \
            + lapl_part + abs_part

    def _gradient_p(self):
        """
        Calculate the gradient of the loss function with respect to p
        :return: gradient
        """
        # no need to to the expensive calculations if mu_fro is 0
        if self.mu_fro != 0.:
            # Gradient of the Frobenius part : (G^T)GP
            if self.first_column_fro:
                fro_part = self.mu_fro * (self.g_matr.T @ self.d_matr)
                fro_part[:,0] = 0.
            else:
                fro_part = self.mu_fro * (self.g_matr.T @ self.d_matr)
        else:
            fro_part = 0.

        # gradient = - G^T(X + GPA)A^T
        return - np.linalg.multi_dot((self.g_matr.T, self.x_matr, self.a_matr.T)) + \
            np.linalg.multi_dot((self.g_matr.T, self.d_matr, self.a_matr, self.a_matr.T)) + fro_part

    def _f_a_quadr(self, a_tilde, l, grad_a):
        """
        Calculates the proximal approximation with respect to A of the loss function
        :param a_tilde: point in which the loss function is approximated
        :param l: upper bound of the lipschitz constant
        :param grad_a: the gradient with respect to A of the loss function, evaluated in the working point
        :return: the proximal approximation of the loss function
        """
        return self._eval_function() + np.einsum("ij,ij->", grad_a, a_tilde - self.a_matr) \
            + l / 2 * np.linalg.norm(a_tilde - self.a_matr) ** 2

    def _f_p_quadr(self, p_tilde, l, grad_p):
        """
        Calculates the proximal approximation with respect to P of the loss function
        :param p_tilde: point in which the loss function is approximated
        :param l: upper bound of the lipschitz constant
        :param grad_p: the gradient with respect to P of the loss function, evaluated in the working point
        :return: the proximal approximation of the loss function
        """
        return self._eval_function() + np.einsum("ij,ij->", grad_p, p_tilde - self.p_matr) \
               + l / 2 * np.linalg.norm(p_tilde - self.p_matr) ** 2

    # make a step for optimizing A
    def _make_step_a(self):
        """
        Take a gradient descent step in A
        The step size is calculated using the line-search algorithm
        """
        # calculate the gradient with respect to A, in the current point: this
        # will remain constant throughout the line-search
        grad_a = self._gradient_a()

        # calculate the gradient update step based on the previous eta_A estimate
        a_tilde = self._project_on_simplex(self.a_matr - 1 / self.eta_a * grad_a)

        # iteratively increase the eta_A parameter until the criterion below is met
        while self._f_a_quadr(a_tilde, self.eta_a, grad_a) < self._eval_function(a_matr=a_tilde):
            self.eta_a *= self.beta
            a_tilde = self._project_on_simplex(self.a_matr - 1 / self.eta_a * grad_a)

        # if the eta_a value is only increased, then it could possibly be unnecessary small at the end
        # of the algorithm, thus it is also decreased after every line-search
        self.eta_a /= self.alpha

        # perform the actual update of the activations matrix
        self.a_matr = a_tilde

        # update the cached variable AL
        if self.mu_lapl != 0.:
            self.a_dot_l = self.a_matr @ self._laplacian_a

        # this can be very useful to debug issues with the algorithm: to big eta_A values
        # will cause the algorithm to take very, very small steps and thus not converge
        if self.debug:
            self.eta_as.append(self.eta_a)

    def _make_step_p(self):
        """
        Take a gradient descent step in P
        The step size is calculated using the line-search algorithm
        """
        # calculate the gradient with respect to P, in the current point: this
        # will remain constant throughout the line-search
        grad_p = self._gradient_p()

        # calculate the gradient update step based on the previous eta_P estimate
        p_tilde = (self.p_matr - 1 / self.eta_p * grad_p).clip(min=0)

        # iteratively increase the eta_A parameter until the criterion below is met
        while self._f_p_quadr(p_tilde, self.eta_p, grad_p) < self._eval_function(p_matr=p_tilde):
            self.eta_p *= self.beta
            p_tilde = (self.p_matr - 1 / self.eta_p * grad_p).clip(min=0)

        # if the eta_p value is only increased, then it could possibly be unnecessary small at the end
        # of the algorithm, thus it is also decreased after every line-search
        self.eta_p /= self.alpha

        # perform the actual update of the activations matrix
        self.p_matr = p_tilde

        # update the cached variable D
        self.d_matr = self.g_matr @ self.p_matr

        # this can be very useful to debug issues with the algorithm: to big eta_P values
        # will cause the algorithm to take very, very small steps and thus not converge
        if self.debug:
            self.eta_ps.append(self.eta_p)

    def _initialize(self, x_matr):   #TODO Modify description
        """
        Initialize all the mdata used in the algorithm:
        - Flatten and store the data matrix X in the class, also store the original shape
        - Randomly initialize the activations matrix A
        - Initialize the peaks matrix P to 0
        - Create the laplacian matrix for A
        - Create the 'cached' variables: A L, trace(X^T X), G P
        - Set the number of iterations equal to 0
        - Reset the debugging/convergence metrics
        :param x_matr: input data matrix X (as a NxMxK matrix, with K number of energy channels)
        """
        # store the original shape of the input data X
        self.x_shape = x_matr.shape
        # flatten X to a Kx(NM) matrix, such that the columns hold the raw spectra
        x_matr = x_matr.reshape((self.x_shape[0] * self.x_shape[1], self.x_shape[2])).T  # flatten X
        self.x_matr = x_matr.astype(np.float)
        
        if self.initialize_matrix:
            # Initialize randomly the activation matrix but with the first column to ones
            self.a_matr = self._project_on_simplex(np.random.rand(self.p_, self.x_matr.shape[1]))
            self.a_matr[0] = np.ones(self.a_matr.shape[1])
            self.a_matr = self._project_on_simplex(self.a_matr)
            
            # Initialize randomly the peek matrix
            self.p_matr = np.random.rand(self.g_matr.shape[1], self.p_)

            # Set the first spectra as the average spectra
            avg_spectra = x_matr.mean(axis=1)
            linear_reg = LinearRegression().fit(self.g_matr,avg_spectra)
            self.p_matr[:,0] = linear_reg.coef_.clip(min=0)

        else:
            # initialize the activations matrix randomly
            self.a_matr = self._project_on_simplex(np.random.rand(self.p_, self.x_matr.shape[1]))

            # initialize the peaks matrix to zeros
            self.p_matr = np.random.rand(self.g_matr.shape[1], self.p_)

        # create the laplacian matrix if necessary
        if self.mu_lapl != 0.:
            self._laplacian_a = self._create_laplacian_a(self.x_matr.shape[1], self.x_shape[1])
            self.a_dot_l = self.a_matr @ self._laplacian_a

        # create the cached variables
        self.tr_x_x = np.einsum("ij,ij->", self.x_matr, self.x_matr)
        self.d_matr = self.g_matr @ self.p_matr

        self.num_iterations = 0

        # for debugging purposes
        self.eta_as = []
        self.eta_ps = []
        self.base_losses = []

        # to observe the convergence of the algorithm
        self.losses = []
        self.p_update = []
        self.a_update = []
        self.a_norm = []
        self.p_norm = []

    def _iteration(self, step_a=True, step_p=True):
        """
        Execute 1 iteration of the proximal alternating gradient descent algorithm
        :param step_a: whether to take a step in A (for debugging purposes)
        :param step_p: whether to take a step in P (for debugging purposes)
        :return: loss after taking a step in A, loss after taking a step in P
        """
        # take step in A and evaluate
        if step_a:
            self._make_step_a()
        eval_after_a = self._eval_function()

        # take step in P and evaluate
        if step_p:
            self._make_step_p()
        eval_after_p = self._eval_function()

        self.num_iterations += 1

        return eval_after_a, eval_after_p

    def fit(self, x_matr, max_iter, tol_function, initialize=True, initialize_matrix=False, first_column_fro=False, step_a=True, step_p=True):
        """
        Run the optimization algorithm to fit the input matrix, until either a maximum number
        of iterations is reached (max_iter) or the loss function has decreased by less than
        tol_function
        :param x_matr: input data as a NxMxK matrix (K is the number of energy channels
        :param max_iter: maximum number of iterations to run
        :param tol_function: if the loss function decreases by less than tol_function,
        the optimization algorithm is finished
        :param initialize: whether to initialize all the data stored by the algorithm,
        disabling this allows you to further 'fine tune' the current solution, instead of
        starting all over again
        :param step_a: whether to take a step in A (for debugging purposes)
        :param step_p: whether to take a step in P (for debugging purposes)
        :return: the P matrix, the A matrix
        """

        #set the parameters
        self.initialize_matrix = initialize_matrix
        self.first_column_fro = first_column_fro

        if initialize:
            self._initialize(x_matr)

        eval_after_p = self._eval_function()

        algo_start = time.time()
        while True:
            start = time.time()

            eval_before = eval_after_p

            eval_after_a, eval_after_p = self._iteration(step_a=step_a, step_p=step_p)

            # store some information for assessing the convergence
            self.a_update.append(-(eval_after_a - eval_before))
            self.p_update.append(-(eval_after_p - eval_after_a))
            self.a_norm.append(np.linalg.norm(self.a_matr))
            self.p_norm.append(np.linalg.norm(self.p_matr))
            self.base_losses.append(self._base_loss())
            self.losses.append(eval_after_p)

            # check convergence criterions
            if self.num_iterations >= max_iter:
                print('\nexits because max_iteration was reached')
                break

            elif abs(eval_before - eval_after_p) < tol_function:
                print('\nold function: {}, new_function: {}'.format(str(eval_before), eval_after_p))
                print('exit because of tol_function')
                break

            print(f"\rFinished iteration {self.num_iterations} of maximal {max_iter} function value "
                  f"decreased by: {eval_before-eval_after_p} taking: {time.time()-start} seconds",
                  end="", flush=True)

        self.algo_time = time.time() - algo_start
        algo_time = self.algo_time
        print(f"\nStopped after {self.num_iterations} iterations in {algo_time//60} minutes "
              f"and {np.round(algo_time) % 60} seconds.")

        self.plot_convergence()

        return self.p_matr, self.a_matr

    def plot_convergence(self):
        """
        Helper function to plot the convergence curves of the algorithm
        """
        fig1 = plt.figure(figsize=(15, 3))

        ax1 = fig1.add_subplot(1, 6, 1)
        ax1.plot(self.losses)
        ax1.set_title('F(A,D)')
        ax2 = fig1.add_subplot(1, 6, 2)
        ax2.plot(np.maximum(np.array(self.losses[:-2]) - self.losses[-1], 0.01))
        ax2.set_yscale('log')
        ax2.set_title('F(A,D)_t - F(A,D)_T')
        ax3 = fig1.add_subplot(1, 6, 3)
        ax3.plot(self.a_update)
        ax3.set_yscale('log')
        ax3.set_title('A update')
        ax4 = fig1.add_subplot(1, 6, 4)
        ax4.plot(self.p_update)
        ax4.set_yscale('log')
        ax4.set_title('D update')
        ax6 = fig1.add_subplot(1, 6, 5)
        ax6.plot(self.a_norm)
        ax6.set_title('A_Norm')
        ax7 = fig1.add_subplot(1, 6, 6)
        ax7.plot(self.p_norm)
        ax7.set_title('D_Norm')
        fig1.tight_layout()
        plt.show()
