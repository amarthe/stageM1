import numpy as np
import json
import os
from math import sqrt, pi
from scipy.special import erf


class Gaussians:
    """
    Contains utility functions to handle energy spectra composed of Gaussians
    """

    XRAYS_FILE = os.path.join(os.path.dirname(__file__), "Data", "xrays.json")

#    def __init__(self, width=.136, e_offset=-0.47195, e_size=2048, e_scale=.01):
    def __init__(self, width=.136, e_offset=0.20805000000000007, e_size=1980, e_scale=.01):
        self.x = np.arange(e_offset, e_offset+e_size*e_scale, e_scale)
        self.width = width

        with open(Gaussians.XRAYS_FILE, "r") as f:
            self.xrays = json.load(f)["table"]

    def create_spectrum(self, elements):
        """
        Create an 'ideal' Gaussian spectrum for the provided elements
        :param elements: dictionary mapping an element, to the concentration
        :return: spectrum
        """
        y = np.zeros(self.x.shape)
        for element in self.xrays:
            conc = elements.get(element["element"])
            if conc is not None:
                for i, energy in enumerate(element["energies"]):
                    y += conc * element["ratios"][i] * Distributions.pdf_gaussian(energy,
                                                                                  self.width/2,
                                                                                  self.x)
        return y

    def create_spectrum_decomp(self, elements):
        """
        Create an array of peak values of the Gaussian peaks in the spectrum
        :param elements: dictionary mapping an element, to the concentration
        :return: array of peaks
        """
        y = []
        for element in sorted(self.xrays, key=lambda x: x["element"]):
            conc = elements.get(element["element"].split("-")[0], 0.)
            y.append(conc)

        return np.array(y)

    def create_matrix(self):
        """
        Creates a matrix containing all possible Gaussians (based on the elements in the xrays file
        """
        g_matr = np.zeros((self.x.size, 0))
        for element in sorted(self.xrays, key=lambda x: x["element"]):
            signal = np.zeros((self.x.size, 1))
            for i, energy in enumerate(element["energies"]):
                signal += element["ratios"][i] * Distributions.pdf_gaussian(energy,
                                                                            self.width/2,
                                                                            self.x)[np.newaxis].T
            g_matr = np.concatenate((g_matr, signal), axis=1)
        return g_matr


class Distributions:
    @staticmethod
    def pdf_gaussian(mu, sigma, scale):
        return np.exp(-np.power(scale - mu, 2) / (2 * np.power(sigma, 2)))
    
    @staticmethod
    def eggert_brstlg(a0,a1,a2,mu,scale) :
        return (a0*((100-scale)/scale)+a1*((100-scale)**2/scale))*np.exp(mu/np.power(scale,a2))

    @staticmethod
    def _cdf_log_normal(mus, sigmas, scale):
        """
        Cumulative distribution function of the Log Normal Distribution
        :param mus: array of means (or single mean)
        :param sigmas: array of std's (or single std)
        :param scale: the x scale
        :return: array with the cum distr for every (mu, sigma) pair
        """
        with np.errstate(divide='ignore'):
            return .5 + .5 * erf((np.log(scale) - mus) / sqrt(2) / sigmas)

    @staticmethod
    def _cdf_log_normal_dmu(mus, sigmas, scale):
        """
        Derivative with respect to mu of the cumulative distribution function of the Log Normal Distribution
        :param mus: array of means (or single mean)
        :param sigmas: array of std's (or single std)
        :param scale: the x scale
        :return: array with the derivative of the cum distr for every (mu, sigma) pair
        """
        with np.errstate(divide='ignore'):
            return -1 / sqrt(2*pi) / sigmas * np.exp(-1 / 2 / sigmas**2 * (np.log(scale) - mus) ** 2)

    @staticmethod
    def _cdf_log_normal_dsigma(mus, sigmas, scale):
        """
        Derivative with respect to sigma of the cumulative distribution function of the Log Normal Distribution
        :param mus: array of means (or single mean)
        :param sigmas: array of std's (or single std)
        :param scale: the x scale
        :return: array with the derivative of the cum distr for every (mu, sigma) pair
        """
        with np.errstate(divide='ignore'):
            if scale[0, 0] == 0:
                return np.concatenate((np.zeros((1, len(mus))), - (np.log(scale[1:]) - mus) / sqrt(2*pi) / sigmas**2 *
                                       np.exp(-1 / 2 / sigmas**2 * (np.log(scale[1:]) - mus)**2)), axis=0)
            else:
                return - (np.log(scale) - mus) / sqrt(2 * pi) / sigmas ** 2 * \
                    np.exp(-1 / 2 / sigmas ** 2 * (np.log(scale) - mus) ** 2)

    @staticmethod
    def pdf_log_normal(mus, sigmas, scale):
        """
        Density function of the Log Normal Distribution
        :param mus: array of means (or single mean)
        :param sigmas: array of std's (or single std)
        :param scale: the x scale
        :return: array with the density for every (mu, sigma) pair
        """
        cdf = Distributions._cdf_log_normal(mus, sigmas, scale)
        return cdf[1:] - cdf[:-1]

    @staticmethod
    def pdf_log_normal_dmu(mus, sigmas, scale):
        """
        Derivative with respect to mu of the density function of the Log Normal Distribution
        :param mus: array of means (or single mean)
        :param sigmas: array of std's (or single std)
        :param scale: the x scale
        :return: array with the derivative of the density for every (mu, sigma) pair
        """
        cdf = Distributions._cdf_log_normal_dmu(mus, sigmas, scale)
        return cdf[1:] - cdf[:-1]

    @staticmethod
    def pdf_log_normal_dsigma(mus, sigmas, scale):
        """
        Derivative with respect to sigma of the density function of the Log Normal Distribution
        :param mus: array of means (or single mean)
        :param sigmas: array of std's (or single std)
        :param scale: the x scale
        :return: array with the derivative of the density for every (mu, sigma) pair
        """
        cdf = Distributions._cdf_log_normal_dsigma(mus, sigmas, scale)
        return cdf[1:] - cdf[:-1]


class MatrixUtils:
    """
    Some utility matrix operations
    """
    @staticmethod
    def shift_matrix_vert(matr, k):
        """
        Shift a matrix by k rows
        """
        if k >= 0:
            return np.concatenate((np.zeros((k, *matr.shape[1:])), matr[:-k]), axis=0)
        else:
            k = -k
            return np.concatenate((matr[k:], np.zeros((k, *matr.shape[1:]))), axis=0)

    @staticmethod
    def shift_matrix_horiz(matr, k):
        """
        Shift a matrix by k columns
        """
        if k >= 0:
            return np.concatenate((np.zeros((matr.shape[0], k, *matr.shape[2:])), matr[:, :-k]), axis=1)
        else:
            k = -k
            return np.concatenate((matr[:, k:], np.zeros((matr.shape[0], k, *matr.shape[2:]))), axis=1)

    @staticmethod
    def shift_matrix_diag(matr, k):
        """
        Shift a matrix by k rows and k columns
        """
        if k >= 0:
            res = np.concatenate((np.zeros((k, matr.shape[1]-k, *matr.shape[2:])), matr[:-k, :-k]), axis=0)
            return np.concatenate((np.zeros((matr.shape[0], k, *matr.shape[2:])), res), axis=1)
        else:
            k = -k
            res = np.concatenate((matr[k:, k:], np.zeros((k, matr.shape[1]-k, *matr.shape[2:]))), axis=0)
            return np.concatenate((np.zeros((matr.shape[0], k, *matr.shape[2:])), res), axis=1)
        
class MetricsUtils :

    @staticmethod
    def spectral_angle(v1, v2):
        v1_u = v1/np.linalg.norm(v1)
        v2_u = v2/np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

    @staticmethod
    def MSE_map(map1,map2) :
        tr_m1_m1=np.einsum("ij,ij->", map1, map1)
        tr_m2_m2=np.einsum("ij,ij->", map2, map2)
        tr_m1_m2=np.trace(map1.T@map2)
        return tr_m1_m1 - 2*tr_m1_m2 + tr_m2_m2
