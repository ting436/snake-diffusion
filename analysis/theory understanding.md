Theory
1. Reason of appearing of diffusion models. [Main paper](https://arxiv.org/pdf/1503.03585):
    - Probabilistic suffer from tradeoff: tractability and flexibility. Tractability is gaussian distribution, which is easy to track but hard to describe with it rich datasets. Flexibility can describe complex data structures, but computing hard. For example we can define φ(x) yielding the flexible distribution $p(x) = φ(x) / Z $, where Z is a normalization constant. But Z is hardly to compute, because $Z = \int φ(x) dx$, where φ(x) - complex and high dimensional function. Solution: Markov chain where every step of it is tractable
    - Forward Trajectory:
      $ \pi(y) = \int T_\pi(y|y'; \beta) \pi(y') dy'$

      $\pi(y)$ - final distribution, $T_\pi(y|y'; \beta)$ - diffusion kernel

      $q(x^{(t)}|x^{(t-1)}) = T_\pi(x^{(t)}|x^{(t-1)};\beta_t)$

      $q(\mathbf{x}^{(0...T)}) = q(\mathbf{x}^{(0)})\prod_{t=1}^T q(\mathbf{x}^{(t)}|\mathbf{x}^{(t-1)})$
2. Diffusion models (Sohl-Dickstein et al., 2015) are a class of generative models inspired by nonequilibrium thermodynamics that generate samples by reversing a noising process. -> [Main paper](https://arxiv.org/pdf/1503.03585)
3. This diffusion process can be described as the solution to a standard stochastic differential equation
(SDE) (Song et al., 2020).

    - Formula: $dx = \mathbf{f}(\mathbf{x},\tau)d\tau + g(\tau)d\mathbf{w}$ . [Original paper Stohastic differential Equation](https://arxiv.org/pdf/2011.13456), which depends on [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600), which introduced Score based generation $\nabla_x \log p(x)$, which originally described in [Kernel Stein disperancy](https://arxiv.org/pdf/1602.03253) with a lot of hard math

    - About Score Based Generation from [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600):

      [Source code](https://github.com/ermongroup/ncsn/tree/master)

      $\nabla_x \log p(x)$ - score, calculated with $E_{p_v}E_{p_data}[v^T∇_x s_θ(x)v + \frac{1}{2}||s_θ(x)||_2^2]$. Without Noise, it's performed bad and with noise it works great.

      Also they 

      Then they introduced Noise Conditional Score Network:

      > Let ${\sigma_i}^L_{i=1}$ be a positive geometric sequence that satisfies $\frac{\sigma_1}{\sigma_2} = \cdots = \frac{\sigma_{L-1}}{\sigma_L} > 1$. Let $q_\sigma(\mathbf{x}) \triangleq \int p_{\text{data}}(\mathbf{t})\mathcal{N}(\mathbf{x}|\mathbf{t},\sigma^2I)d\mathbf{t}$ denote the perturbed data distribution. We choose the noise levels ${\sigma_i}^L_{i=1}$ such that $\sigma_1$ is large enough to mitigate the difficulties discussed in Section 3, and $\sigma_L$ is small enough to minimize the effect on data. We aim to train a conditional score network to jointly estimate the scores of all perturbed data distributions, i.e., $\forall \sigma \in {\sigma_i}^L_{i=1}: s_\theta(\mathbf{x},\sigma) \approx \nabla_\mathbf{x} \log q_\sigma(\mathbf{x})$. Note that $s_\theta(\mathbf{x},\sigma) \in \mathbb{R}^D$ when $\mathbf{x} \in \mathbb{R}^D$. We call $s_\theta(\mathbf{x},\sigma)$ a Noise Conditional Score Network (NCSN)

      > Both sliced and denoising score matching can train NCSNs. We adopt denoising score matching as it is slightly faster and naturally fits the task of estimating scores of noise-perturbed data distributions. However, we emphasize that empirically sliced score matching can train NCSNs as well as denoising score matching. We choose the noise distribution to be $q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})=\mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x},\sigma^2I)$; therefore $\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})=-(\tilde{\mathbf{x}}-\mathbf{x})/\sigma^2$. For a given $\sigma$, the denoising score matching objective (Eq. (2)) is
      >
      > $\ell(\theta;\sigma) \triangleq \frac{1}{2}\mathbb{E}{p{\text{data}}(\mathbf{x})}\mathbb{E}{\tilde{\mathbf{x}}\sim\mathcal{N}(\mathbf{x},\sigma^2I)}\left[\left|s\theta(\tilde{\mathbf{x}},\sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\right|_2^2\right]$. (5)
      > 
      > Then, we combine Eq. (5) for all $\sigma \in {\sigma_i}^L_{i=1}$ to get one unified objective
      >
      > $\mathcal{L}(\theta;{\sigma_i}^L_{i=1}) \triangleq \frac{1}{L}\sum^L_{i=1}\lambda(\sigma_i)\ell(\theta;\sigma_i)$, (6)
      >
      > where $\lambda(\sigma_i)>0$ is a coefficient function depending on $\sigma_i$. Assuming $s_\theta(\mathbf{x},\sigma)$ has enough capacity, $s_{\theta^*}(\mathbf{x},\sigma)$ minimizes Eq. (6) if and only if $s_{\theta^*}(\mathbf{x},\sigma_i)=\nabla_\mathbf{x}\log q_{\sigma_i}(\mathbf{x})$ a.s. for all $i\in{1,2,\cdots,L}$, because Eq. (6) is a conical combination of $L$ denoising score matching objectives.
      > 
      > There can be many possible choices of $\lambda(\cdot)$. Ideally, we hope that the values of $\lambda(\sigma_i)\ell(\theta;\sigma_i)$ for all ${\sigma_i}^L_{i=1}$ are roughly of the same order of magnitude. Empirically, we observe that when the score networks are trained to optimality, we approximately have $|s_\theta(\mathbf{x},\sigma)|2\propto 1/\sigma$. This inspires us to choose $\lambda(\sigma)=\sigma^2$. Because under this choice, we have $\lambda(\sigma)\ell(\theta;\sigma)=\sigma^2\ell(\theta;\sigma)=\frac{1}{2}\mathbb{E}[|\sigma s\theta(\tilde{\mathbf{x}},\sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma}|2^2]$. Since $\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma}\sim\mathcal{N}(0,I)$ and $|\sigma s\theta(\mathbf{x},\sigma)|_2\propto 1$, we can easily conclude that the order of magnitude of $\lambda(\sigma)\ell(\theta;\sigma)$ does not depend on $\sigma$.


4. [Original paper Stohastic differential Equation](https://arxiv.org/pdf/1907.05600) introduced score matching: $
\frac{1}{2}\mathbb{E}_{p_{data}}[\|s_\theta(\mathbf{x}) - \nabla_\mathbf{x}\log p_{data}(\mathbf{x})\|_2^2]$ and for denoising: $\frac{1}{2}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})p_{data}(\mathbf{x})}[\|s_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x})\|_2^2]$

  Sampling via Lavgenin Dynamic: $\tilde{\mathbf{x}}_t = \tilde{\mathbf{x}}_{t-1} + \frac{\epsilon}{2}\nabla_\mathbf{x}\log p(\tilde{\mathbf{x}}_{t-1}) + \sqrt{\epsilon}\mathbf{z}_t$

  Loss function for $q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x},\sigma^2I)$; $\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}|\mathbf{x}) = -\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}$:
  
   $\ell(\theta;\sigma) \triangleq \frac{1}{2}\mathbb{E}_{p_{data}(\mathbf{x})}\mathbb{E}_{\tilde{\mathbf{x}}\sim\mathcal{N}(\mathbf{x},\sigma^2I)}\left[\left\|s_\theta(\tilde{\mathbf{x}},\sigma) + \frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\right\|_2^2\right]$

  $\mathcal{L}(\theta;\{\sigma_i\}_{i=1}^L) \triangleq \frac{1}{L}\sum_{i=1}^L\lambda(\sigma_i)\ell(\theta;\sigma_i)$

5. [Original paper Stohastic differential Equation](https://arxiv.org/pdf/2011.13456)

Introduced: $d\mathbf{x} = \mathbf{f}(\mathbf{x},t)dt + g(t)d\mathbf{w}$ and sampling and training. Need to think about it

6. Original Paper diamond:

$\mathcal{L}(\theta) = \mathbb{E} [\|S_\theta(\mathbf{x}^T, \tau) - \nabla_{\mathbf{x}^T} \log p^{0T}(\mathbf{x}^T|\mathbf{x}^0)\|^2]$

For normal distribution
$\mathcal{L}(\theta) = \mathbb{E} [\|\mathbf{D}_\theta(\mathbf{x}^T, \tau) - \mathbf{x}^0\|^2]$
where $\mathbf{D}_\theta(\mathbf{x}^T, \tau) = S_\theta(\mathbf{x}^T, \tau)\sigma^2(\tau) + \mathbf{x}^T$ (It was in point 4)

Introduced $p^{0\tau}(x_{i+1}^T | x_{i+1}^0) = \mathcal{N}(x_{i+1}^T; x_{i+1}^0, \sigma^2(\tau)I)$, where $g(\tau) = \sqrt{2\dot{\sigma}(\tau)\sigma(\tau)}$ and $\mathbf{f}(\mathbf{x}, \tau) = \mathbf{0}$

[Karras EDM](https://arxiv.org/pdf/2206.00364)

Leads to

$\mathbf{D}_\theta(x_{i+1}^T, y_i^T) = c_{\text{skip}}^{\tau} x_{i+1}^T + c_{\text{out}}^{\tau} \mathbf{F}_\theta(c_{\text{in}}^{\tau} x_{i+1}^T, y_i^T)$

$c_{\text{skip}}^{\tau} = \sigma_{\text{data}}^2/(\sigma_{\text{data}}^2 + \sigma^2(\tau))$

$y_i^T := (c_{\text{noise}}^{\tau}, x_{i}^0, a_{<i})$




$\mathcal{L}(\theta) = \mathbb{E} \left\|\underbrace{\mathbf{F}_\theta(c_{\text{in}}^{\tau}x_{i+1}^T, y_i^T)}_{\text{Network prediction}} - \underbrace{\frac{1}{c_{\text{out}}^{\tau}}(x_{i+1}^0 - c_{\text{skip}}^{\tau}x_{i+1}^T)}_{\text{Network training target}}\right\|^2$

What's plan?
1. Understand on toy examples how ddpm depends on T
2. Understand on toy examples how to calculate score $\nabla_x \log p(x)$ - score, calculated with $E_{p_v}E_{p_data}[v^T∇_x s_θ(x)v + \frac{1}{2}||s_θ(x)||_2^2]$ and how it depends on data
3. EDM - how it draw a formula $\mathbf{D}_\theta(x_{i+1}^T, y_i^T) = c_{\text{skip}}^{\tau} x_{i+1}^T + c_{\text{out}}^{\tau} \mathbf{F}_\theta(c_{\text{in}}^{\tau} x_{i+1}^T, y_i^T)$
4. Rewrite DDPM

After 3 epoch context destroys after 7 epoch more context

Questions and thoughts:
1. DDIM is understandble, why do they use cumolative alphas ?
2. Look at the proof of the math
3. SDE - just formula which contains log of distribution(which means score of its disctribution) and inference with Lavgenengin Dynamics. 
4. How to get this formula ? $\mathbb{E}_{p_{\text{data}}}\left[\text{tr}(\nabla_\mathbf{x}\mathbf{s}_\theta(\mathbf{x})) + \frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2_2\right]$