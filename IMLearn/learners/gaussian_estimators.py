from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    mu_ = 0
    var_ = 0
    fitted_ = ""

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        cur_sum = 0
        # sum the left value in the formula.
        for i in range(len(X)):
            cur_sum += (X[i] - self.mu_) ** 2
        # get 1/m-1*sum
        self.var_ = (1 / (len(X) - 1)) * cur_sum
        self.fitted_ = True
        return self

    def __single_sample_pdf(self, x: float) -> float:
        """
        A short helper method that calculate a single value of the
        UnivariateGaussian pdf, this will be called for each sample
        to calculate the pdf.

        x: sample, flot64
        return: pdf value, float64
        """
        const = 1 / np.sqrt(2 * np.pi * self.var_)
        # the const which is 1/sqrt(2*pi*var)
        exp_pow = (-1 / (2 * self.var_)) * (
                (x - self.mu_) ** 2)  # exp pow in pdf.
        return const * np.exp(exp_pow)  # combine and return

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling"
                             " `pdf` function")

        ret_val = np.ndarray(shape=(len(X),))
        for i in range(len(X)):
            ret_val[i] = self.__single_sample_pdf(X[i])
        return ret_val

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model
        assumption was add that the X samples are iid - as seen in the lecture.

        after simplification we will get the following formula which will get
        calculated
            -m/2*log(2*pi*sigma)-1/(2*sigma)*SUM((x_i - mu)^2)

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        exp_val = 0
        n = X.shape[0]
        for i in range(n):
            exp_val += pow((X[i] - mu), 2)
        exp_val = (1 / (2 * sigma)) * exp_val
        # here we calculate the second element.
        first_val = (-n / 2) * np.log(2 * np.pi * sigma)
        return first_val - exp_val


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    mu_ = ""
    cov_ = ""
    fitted_ = ""

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        n_samples, n_features = X.shape
        self.mu_ = np.ndarray(shape=(n_features,))
        for i in range(n_features):
            cur_col = X[:, i]
            self.mu_[i] = np.mean(cur_col)
            # mean each col in the sample matrix to create mu vector.
        # creates mu matrix so we could centered X
        mu_mat = np.array([self.mu_ for _ in range(n_samples)])
        centered_X = np.subtract(X, mu_mat)
        # use the formula we saw in the book 1/m-1*X~^T*X~
        self.cov_ = np.multiply(
            np.matmul(np.transpose(centered_X), centered_X),
            (1 / (n_samples - 1)))
        self.fitted_ = True
        return self

    def __calculate_pdf_for_single_sample(self, x: np.ndarray, cov_det: float)\
            -> float:
        """
        Helper method to calculate the pdf for a single features vector.

        @param: x, np.ndarray of shape (n_features,)
        @param: cov_det, covariance matrix det

        """
        centered_vec = np.subtract(x, self.mu_)
        # calculate the const at the start of the formula.
        const = 1 / np.sqrt(((2 * np.pi) ** x.shape[0]) * cov_det)
        left_mal_calc = np.matmul(np.transpose(centered_vec),
                                  np.linalg.inv(self.cov_))
        exp_pow = np.multiply(np.matmul(left_mal_calc, centered_vec), -1 / 2)
        return const * np.exp(exp_pow)

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        ret_val = np.ndarray(shape=(X.shape[0],))
        cov_det = np.linalg.det(self.cov_)
        for j in range(X.shape[0]):
            # calc outside then assign the value back
            ret_val[j] = self.__calculate_pdf_for_single_sample(X[j], cov_det)
        return ret_val

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        using the formula that was achieved in the theoretical part which is
         -1/2(m*d*log(2PI)+log(|SIGMA|) + SUM((x_i-mu)^T *SIGMA^-1 * (x_i-mu)))

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under
            given parameters of Gaussian
        """
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        n, d = X.shape
        centered_s = X - mu
        # for j in range(n):
        #     centered_vec = np.subtract(X[j], mu)
        #     # matrix multiplication to get the scalar and sum it.
        #     left = np.matmul(np.transpose(centered_vec), cov_inv)
        #     sum_for_vec += np.matmul(left, centered_vec)
        sum_for_vec = np.sum(centered_s @ cov_inv * centered_s)
        # calc the log sum as seen in the formula.
        log_sum = n * d * np.log(2 * np.pi) + np.log(cov_det)
        return (-1 / 2) * (log_sum + sum_for_vec) # add everything and return
