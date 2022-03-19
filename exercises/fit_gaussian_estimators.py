import tqdm

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    real_mu, real_var = 10, 1
    samples = np.random.normal(10, 1, 1000)
    uni_g = UnivariateGaussian()
    # Question 1 - Draw samples and print fitted model
    uni_g.fit(samples)
    print(f"({uni_g.mu_}, {uni_g.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    x = []
    y = []
    for i in range(10, 1010, 10):
        x.append(i)
        uni_g.fit(samples[:i])  # get only i samples and fit them
        y.append(np.abs(uni_g.mu_ - real_mu))
    go.Figure(go.Scatter(x=x, y=y, mode='markers+lines',
                         name=r'$\widehat\mu$'),
              layout=go.Layout(
                  title=r"$\text{Q2 Particle section - mean's empirical "
                        r"properties are consistent}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="$\widehat\mu\\text{ - absolute distance "
                              "between means}$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_vals = uni_g.pdf(samples)
    go.Figure(go.Scatter(x=samples, y=pdf_vals, mode='markers',
                         name=r'$\widehat\mu$'),
              layout=go.Layout(
                  title=r"$\text{Q3 Particle section - Empirical PDF of "
                        r"fitted model}$",
                  xaxis_title="$\\text{Samples}$",
                  yaxis_title="$PDF values$")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    Sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, Sigma, 1000)
    mg = MultivariateGaussian()
    mg.fit(samples)
    # prints values.
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation
    f_arr = np.linspace(-10, 10, 200)
    # results matrix, which is 200x200
    log_l_vals = np.ndarray(shape=(len(f_arr), len(f_arr)))
    for i in range(len(f_arr)):
        for j in range(len(f_arr)):
            new_mu = np.array([f_arr[i], 0, f_arr[j], 0])
            log_l_vals[i][j] = mg.log_likelihood(new_mu, Sigma, samples)
    go.Figure(go.Heatmap(x=f_arr, y=f_arr, z=np.transpose(log_l_vals)),
              layout=go.Layout(
        title=r"$\text{Q5 Particle section - Log Likelihood Heatmap}$",
        xaxis_title="$f1\\ values$",
        yaxis_title="$f2\\ values$")).show()

    # Question 6 - Maximum likelihood
    maxarg_model_ind = np.where(log_l_vals == np.amax(log_l_vals))
    # wasn't instructed to print the values, but may easily be carried out.
    maxargs = (f_arr[maxarg_model_ind[0][0]], f_arr[maxarg_model_ind[1][0]])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
