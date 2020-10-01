# wasserstein-gan
Implementation of the [Improved Wasserstein Generative Adversarial Network (GAN)](https://arxiv.org/pdf/1704.00028.pdf) on the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

| ![](docs/generator_train.gif) |
| :-: |
| *Generator outputs during training.* |

### Generative Adversarial Networks (GANs):

A GAN is a neural network model consisting of two parts; a generator and a critic (also called a discriminator depending on the specific GAN model). The generator is trained to produce novel instances of data that look real compared to some reference training dataset. In order to "sample" batches of data from the generator model, inputs proved to the model are actually sampled from a prior distribution (typically Gaussian) which then produces a "sampled" batch of output data. The critic is trained to be able to discern between data produced by the generator ("fake" data) and data selected from the training dataset ("real" data). These two models are trained against each other with competing objectoves, but as one improves, so does the other and at the end of training you get a generator that can produce realistic looking data that "fools" the critic. GAN's are often trained on image datasets in order to produce novel images that look like realistic samples from the training dataset.

### GANs and Probability Distributions:

One way to interperet a dataset is that it consists of observations of a random variable which belongs to some complex probability distribution. Different observations will have different probabilities of occurence. If we have some training dataset consisting of "real" data (i.e. images of dogs), we can say this data belongs to some probability distribution of occurrence in which zero probability is assigned for data outside this distribution (i.e. images of lamps) and non-zero probability is assigned for data within this distribution (i.e. images of golden retrievers).

In a GAN, the critic learns to detect whether input data is sampled from "real" or "fake" probability distributions and the generator learns a function approximation of the "real" data probability distribution using the critics output as a supervision proxy signal.

### GAN Implementations:

In the [original GAN implementation](https://arxiv.org/pdf/1406.2661.pdf), the critic model is actually refered to as the *discriminator* model, as it is trained to classify whether an input image is "fake" (from the generator), or "real" from the training dataset. This approach is successful, but the presence of sigmoid activation at the output of the discriminator can lead to saturation during traing of some complex data distributions, leading to instability and difficulty training in some cases.

The Improved Wasserstein GAN is introduced to remedy some of the instability issues faced by the [original GAN](https://arxiv.org/pdf/1406.2661.pdf) by defining a different objective for the critic. This new objective involves training the critic such that the output can directly be used to compute an estimate of the divergence between the probability distributions in which two batches of input data are sampled from. This is why the name "critic" is used instead of the name "discriminator", as this model now outputs an unbounded real value more analogous to "how real a batch of data is" rather than the probability that a batch of data is "real". In this setting, the critic and an identical copy of the critic are given a batch of "real" data sampled from the training dataset, and a batch of "fake" data sampled from the generator model, respectively. The difference between the output of the critic given "real" data and the output of the critic copy given "fake" data is used as an estimate of the [Wasserstein Distance](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the probability distributions from which the two batches of input data were sampled from. The Wasserstein distance is a probability distribution divergence metric based on [Earth Movers Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance), and can be interpereted loosely as the amount of "work" it would take to convert one probability distribution to another by physically moving "probability density" between the two. If two batches of data are sampled from the same distribution, the Wasserstein distance between these two samples is zero. In contrast, a non-zero Wasserstein distance between two samples of data indicates that these batches of data were sampled from different probability distributions. In the Wasserstein GAN, this property is used to define both the critic and generator objectives, in which the critic learns to maximize a Wasserstein distance estimate (to better differentiate between "real" and "fake" data), and the generator learns to minimize the same Wasserstein distance estimate (to better fool the critic with the generated "fake" data). The magic beind this implementation is that we can simply optimize these objectives without worrying about whether or not the estimated Wasserstein distance has any actual meaningful value! The Wasserstein distance "units" learned by the critic and the generator simply cancel each other out! In the following section we present the mathematical formulations of these objectives, in which we can see in more detail why this is the case.

### Defining the Objective Functions:

For a critic model, $C_\theta$, with parameters $\theta$, "real" input data batch, $x_r$, and "fake" input data batch, $x_f$, the estimated Wasserstein distance estimate, $W$, between the distributions from which the two input batches were sampled is expressed by:

\begin{align}
W(x_r, x_f) = C_\theta(x_r) - C_\theta(x_f).
\tag{1}
\label{eq:wass_dist}
\end{align}

A nice way to remember the Wasserstein distance estimate in terms of critic output is "real minus fake".

Since $x_f$ is sampled from the generator network, we can substitute $x_f = G_\phi(z)$ for generator network, $G$, with parameters $\phi$, and inputs, $z$, sampled from a prior distribution, changing eq. \ref{eq:wass_dist} to:

\begin{align}
W(x_r, z) = C_\theta(x_r) - C_\theta(G_\phi(z)).
\tag{2}
\label{eq:wass_dist_full}
\end{align}

At this point, we can see that $\frac{\partial W}{\partial \theta}$ and $\frac{\partial W}{\partial \phi}$ can both be computed from eq. \ref{eq:wass_dist_full}. Remembering also from above that we want the critic to maximize the Wasserstein distance while we want the generator to minimize Wasserstein distance, this sets us up nicely to define the critic and generator objectives both in terms of eq. \ref{eq:wass_dist_full}.

1. **Critic Objective:** We define the critic objective, $J_c$, as the maximization of eq. \ref{eq:wass_dist_full} by defining the minimzation of the negation:

\begin{align}
J_c = \min_{\theta}\left( -W(x_r, z) \right)
\tag{3}
\label{eq:crit_obj}
\end{align}

2. **Generator Objective:** We define the generator objective, $J_g$, as the minimzation of eq. \ref{eq:wass_dist_full}:

\begin{align}
J_g = \min_{\phi}\left( W(x_r, z) \right)
\tag{4}
\label{eq:gen_obj}
\end{align}

*NOTE:* Since only the $-C_\theta(G_\phi(z))$ term of eq. \ref{eq:wass_dist_full} is dependent on the generator parameters, $\phi$, we should technically define the generator objective as $J_g = \min_{\phi}\left(-C_\theta(G_\phi(z))\right)$. We leave the objective as eq. \ref{eq:gen_obj} in the code for readibility since Pytorch can be configured to automatically ignore specific variables during different update phases.

3. **Critic Gradient Regularizer:** The "Improved" part of the Improved Wasserstein GAN involves a gradient penalty in order to promote a [1-Lipschitz constrain](https://en.wikipedia.org/wiki/Lipschitz_continuity) on the critic output function. This gradient penalty aims to enforce that the critic output gradient w.r.t the inputs has a value of 1 everywhere. In practice, computing this gradient for every possible input is clearly intractable, therefore the gradient is computed for a carefully selected batch of input samples. This sample batch is constructed by randomly interpolating between the batch of "real" data and the batch of "fake" data from the generator. The "sampled gradient" at this interpolated batch of data is pushed to be a value of one with a Lagrangian multiplier regularization term added to the critic objectove function. With the introduction of a randomly interpolated data batch, $x_i$, the "Improved" critic objectove function becomes,

\begin{align}
J_c = \min_{\theta}\left( -W(x_r, z) + \lambda \left( ||\nabla_{\theta} C_\theta(x_i)|| - 1 \right)^2 \right)
\tag{5}
\label{eq:crit_obj_reg}
\end{align}

where $\nabla_{\theta}$ is the gradient operator w.r.t parameters $\theta$ and $\lambda$ is the scalar regularization strength usually set to 10.

### Training:
