# Exact Gaussian Oracles for Diagnosing the Dimension Dependence of Discretized Langevin Samplers: An Early-Stage Testbed Proposal

**Aarush Kandukoori** (Carnegie Mellon University, Lead Student Author)  
**Aditya Kandukoori** (Poolesville High School, Co-Author)

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Recent analyses of midpoint discretizations of Langevin dynamics show that in high dimension deterministic midpoint methods can outperform randomized ones, and that a simple Gaussian calculation gives randomized and Poisson midpoint methods a lower bound of order d^(1/3) ε^(-1/3) [chen2026]. We propose a small open-source testbed of exact Gaussian references for statements of this kind. On a Gaussian target N(0, Σ) every scheme we consider is linear in the state, so its second and fourth moments obey exact finite linear recursions, with no Monte Carlo noise. For the unadjusted Langevin algorithm (ULA) this recovers a known closed form, which we keep as a sanity check. For the overdamped randomized midpoint method we derive the exact stationary covariance by averaging the second-moment recursion over the random midpoint α ~ Unif[0, 1]: the variance in an eigen-direction with precision λ is inflated by the factor ρ(hλ), where ρ(a) = (2 - 2a + a²)/(2 - 2a + a² - a³/3) = 1 + a³/6 + O(a⁴), and the second moments converge if and only if h times the largest precision is below 1.596. Because one α is shared by all coordinates, the stationary law is a Gaussian scale mixture, not a Gaussian. Its covariance therefore gives only exact lower bounds on the KL and W2 bias, which grow like d and √d for isotropic spectra, while exact fourth moments measure the remaining shared-midpoint part, which a second-order expansion predicts to grow like d² in KL. The testbed asks how the two parts depend on the spectrum and the condition number, a question that uniform upper bounds do not resolve. This is a plan only: no experiments have been run, and all data are synthetic. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, and an AI review checked the draft before release, and the authors take responsibility for all content.

## Introduction & Motivation

**Status.** This is an early-stage proposal by two students. It contains no experimental results, and we are seeking expert feedback before building anything substantial.

**Problem.** Non-asymptotic theory for discretized Langevin samplers bounds the step size and number of steps needed to reach accuracy $\varepsilon$ in dimension $d$, uniformly over smooth strongly log-concave targets [dalalyan2017, durmus2017, shen2019, altschuler2026]. Such bounds do not say how a given method behaves on a given target, and Monte Carlo estimates of $W_2$ or KL in high dimension are usually too noisy to resolve small discretization biases. [chen2026] sharpen the picture for midpoint methods: for deterministic Heun and underdamped exponential midpoint schemes, a lower bound matches the $d$ and $\varepsilon$ powers of their upper bounds at bounded condition number, while for randomized and Poisson midpoint methods a simple Gaussian calculation gives a lower bound of order $d^{1/3}\varepsilon^{-1/3}$. So for randomized schemes, Gaussian targets are already informative.

**Idea.** For $\pi=\mathcal N(0,\Sigma)$ every scheme below is linear in the state. Conditional on its random midpoints, a randomized scheme is a linear Gaussian recursion, so its moments can be computed exactly by finite linear recursions, with no sampling noise. We work this out for the overdamped randomized midpoint method and use it to separate two sources of stationary bias: (i) the error in the stationary covariance, and (ii) the non-Gaussianity that arises because one random midpoint is shared by all coordinates.

**Question the testbed answers.** Upper bounds hold uniformly over a class of targets, and lower bounds come from particular hard instances; neither says how the bias on a specific Gaussian target depends on its spectrum. For Gaussian targets with a given spectrum (not only isotropic ones) and condition number $\kappa$, the testbed will compute the exact covariance part of the randomized midpoint bias, the exact fourth-moment excess that drives the shared-midpoint part, and, for isotropic spectra, the full KL and $W_2$ bias through a one-dimensional reduction. This answers three concrete questions: for which spectrum shapes the shared-midpoint part dominates; how the step count implied by the Gaussian bias depends on $\kappa$ as well as on $d$; and how far below the uniform upper bounds the Gaussian bias of each method sits. We make no prediction beyond the derivations below.

**Scope.** We do not propose a new sampler or a new bound. To our knowledge, the contribution would be a reproducible set of exact Gaussian references and the derivations behind them. We expect parts of the derivation to be familiar to experts and would like to learn where they already appear.

## Related Work

- **ULA.** Dalalyan and Durmus-Moulines give non-asymptotic guarantees for the unadjusted Langevin algorithm under smoothness and (strong) log-concavity [dalalyan2017, durmus2017]. On Gaussian targets the ULA stationary law is explicit: [wibisono2018] works out ULA and related algorithms explicitly for Gaussian targets and shows that ULA is not consistent there, while a symmetrized variant is.
- **Randomized midpoint.** Shen and Lee introduce the randomized midpoint method, built on underdamped Langevin dynamics [shen2019]. He, Balasubramanian, and Erdogdu characterize the stationary distribution of randomized midpoint chains, overdamped and underdamped, show that it is biased away from the target, and note that the step size must go to zero for asymptotic unbiasedness [he2020]. Yu, Karagulyan, and Dalalyan give computable $W_2$ upper bounds, building on an analysis of the midpoint discretization of the overdamped Langevin process [yu2023]; the overdamped scheme below is written in their form. Altschuler and Chewi give KL guarantees for the randomized midpoint discretization of the Langevin diffusion, with an $\tilde O(\sqrt d/\varepsilon)$ rate under strong log-concavity or a log-Sobolev inequality [altschuler2026].
- **Midpoint methods and dimension.** Chen et al. give sharp dimension dependence for deterministic midpoint schemes and the Gaussian lower bound for randomized and Poisson midpoint methods described above [chen2026].
- **Metropolis adjustment.** Chewi et al. study the optimal dimension dependence of MALA from a warm start [chewi2021]; MALA enters here only as a simulation reference.
- **Gaussian comparison bound.** For any two distributions with finite second moments, $W_2$ is at least the $W_2$ distance between the Gaussians with the same means and covariances [gelbrich1990].

We have not exhaustively surveyed the literature and welcome pointers, especially to existing exact Gaussian calculations for randomized midpoint methods.

## Proposed Framework

**Notation.** The target is $\pi=\mathcal N(0,\Sigma)$ with $\Sigma\succ0$ and eigen-decomposition $\Sigma=U\,\mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)\,U^\top$, $U$ orthogonal. The precisions are $\lambda_i=\sigma_i^{-2}$, with $m=\min_i\lambda_i$, $L=\max_i\lambda_i$, and condition number $\kappa=L/m$. The potential is $f(x)=\tfrac12x^\top\Sigma^{-1}x$, so $\nabla f(x)=\Sigma^{-1}x$. The step size is $h>0$, and $a_i=h\lambda_i$. $B$ is a standard Brownian motion in $\mathbb R^d$, and $\xi_k\sim\mathcal N(0,I_d)$ are i.i.d. For a covariance matrix $S$, $\gamma_S=\mathcal N(0,S)$. $\mathrm{KL}$ is the Kullback-Leibler divergence, $\chi^2$ the chi-squared divergence, and $W_2$ the 2-Wasserstein distance. $\varepsilon$ denotes a target accuracy.

**Reduction to coordinates.** Every scheme below commutes with the rotation $x\mapsto U^\top x$: $\nabla f$ is linear, Brownian increments are rotation invariant, and the random midpoint $\alpha$ is a single scalar per step, shared by all coordinates. So we may take $\Sigma$ diagonal. Coordinate $i$ then follows a scalar recursion with $\lambda=\lambda_i$, $\sigma^2=\sigma_i^2$, $a=a_i$, and the coordinates interact only through the shared $\alpha$. We drop the index $i$ when only one coordinate is involved.

### Known baseline: ULA

The update $x_{k+1}=x_k-h\nabla f(x_k)+\sqrt{2h}\,\xi_k$ reads $x_{k+1}=(1-a)x_k+\sqrt{2h}\,\xi_k$ in each coordinate. For $hL<2$ the variance recursion $v_{k+1}=(1-a)^2v_k+2h$ has the fixed point
$$v^{\rm ULA}=\frac{2h}{1-(1-a)^2}=\frac{2h}{a(2-a)}=\frac{\sigma^2}{1-a/2},$$
using $h/a=\sigma^2$. The stationary law is the Gaussian $\mathcal N\big(0,\mathrm{diag}(v^{\rm ULA}_i)\big)$, and both laws are centered Gaussians with commuting covariances, so
$$W_2^2=\sum_{i=1}^d\sigma_i^2\Big((1-a_i/2)^{-1/2}-1\Big)^2=\frac{h^2}{16}\sum_{i=1}^d\lambda_i\,\big(1+O(hL)\big),$$
since $(1-a/2)^{-1/2}-1=a/4+O(a^2)$. This standard calculation (compare [wibisono2018]) is our sanity check: the ULA bias is first order in $h$ and, for an isotropic spectrum, proportional to $\sqrt d$ in $W_2$.

### Overdamped randomized midpoint (RMP): exact second moments

**Scheme.** Following the form in [yu2023] (the overdamped version of the construction in [shen2019]), draw $\alpha_k\sim\mathrm{Unif}[0,1]$ independently of everything else and set
$$y_k=x_k-\alpha_kh\,\nabla f(x_k)+\sqrt2\,\big(B_{(k+\alpha_k)h}-B_{kh}\big),\qquad x_{k+1}=x_k-h\,\nabla f(y_k)+\sqrt2\,\big(B_{(k+1)h}-B_{kh}\big).$$
The midpoint and the full step use the same Brownian path. In one coordinate write $W'=B_{(k+\alpha_k)h}-B_{kh}$ and $W=B_{(k+1)h}-B_{kh}$. Given $\alpha_k=\alpha$, the pair $(W',W)$ is centered Gaussian and independent of $x_k$, with $\mathrm{Var}\,W'=\alpha h$, $\mathrm{Var}\,W=h$ and $\mathrm{Cov}(W',W)=\alpha h$.

**Step 1: one step is linear given $\alpha$.** Write $x$, $y$, $x'$ for this coordinate of $x_k$, $y_k$, $x_{k+1}$. With $\nabla f(x)=\lambda x$ in this coordinate, $y=(1-\alpha a)\,x+\sqrt2\,W'$, and then $x'=x-a\,y+\sqrt2\,W=c(\alpha)\,x+\eta$ with
$$c(\alpha)=1-a+\alpha a^2,\qquad \eta=\sqrt2\,(W-aW').$$
Given $\alpha$, $\eta$ is centered Gaussian with variance
$$s^2(\alpha)=2\big(h-2a\,\alpha h+a^2\alpha h\big)=2h\big(1-\alpha a(2-a)\big).$$

**Step 2: second-moment recursion, averaged over $\alpha$.** Let $v_k=\mathbb E[x_k^2]$. Because $\alpha_k$ is independent of $x_k$ and $\mathbb E[\eta\mid\alpha,x_k]=0$, the cross term $2\,\mathbb E[c(\alpha)\,x_k\,\eta]$ vanishes and
$$v_{k+1}=\mathbb E_\alpha\big[c(\alpha)^2\big]\,v_k+\mathbb E_\alpha\big[s^2(\alpha)\big].$$
The average is of $c(\alpha)^2$, not of $c(\alpha)$. With $\mathbb E\alpha=\tfrac12$ and $\mathbb E\alpha^2=\tfrac13$,
$$\mathbb E[c^2]=(1-a)^2+(1-a)a^2+\frac{a^4}{3}=1-2a+2a^2-a^3+\frac{a^4}{3},\qquad \mathbb E[s^2]=h\,(2-2a+a^2).$$
Equivalently $\mathbb E[c^2]=c(\tfrac12)^2+a^4\,\mathrm{Var}(\alpha)=c(\tfrac12)^2+a^4/12$. Plugging $\alpha=\tfrac12$ into the recursion instead describes a different, deterministic rule (treated below).

**Step 3: stability.** Let $p(a)=2-2a+a^2$ and $q(a)=p(a)-a^3/3$. Then $1-\mathbb E[c^2]=a\,q(a)$. Since $q(0)=2$ and $q'(a)=-\big((1-a)^2+1\big)<0$, the recursion contracts if and only if $a<a_\star$, where $a_\star$ is the unique real root of $a^3-3a^2+6a-6=0$ (this cubic is $-3q(a)$), namely $a_\star=1+\sqrt[3]{1+\sqrt2}-\sqrt[3]{\sqrt2-1}\approx1.596$. So the second moments converge in every direction if and only if $hL<a_\star$, compared with $hL<2$ for ULA.

**Step 4: fixed point.** For $a<a_\star$,
$$v=\frac{\mathbb E[s^2]}{1-\mathbb E[c^2]}=\frac{h\,p(a)}{a\,q(a)}=\sigma^2\rho(a),\qquad \rho(a)=\frac{p(a)}{q(a)}=\frac{2-2a+a^2}{2-2a+a^2-a^3/3}=1+\frac{a^3}{6}+\frac{a^4}{6}+O(a^5).$$
Undoing the rotation, the exact stationary covariance is
$$S_h=\Sigma\,\rho\big(h\Sigma^{-1}\big)=U\,\mathrm{diag}\big(\sigma_i^2\,\rho(a_i)\big)\,U^\top,$$
with $\rho$ applied to the matrix through its eigenvalues. Starting from $x_0\sim\pi$, the exact variance after $k$ steps in direction $i$ is $\sigma_i^2\big[\rho(a_i)+\big(1-\rho(a_i)\big)\,\mathbb E[c_i^2]^k\big]$, where $c_i$ is $c$ with $a=a_i$. Since $\rho>1$, the scheme always inflates the variance, by a relative amount $a^3/6+O(a^4)$ against $a/2+O(a^2)$ for ULA.

**Fixed midpoint, for comparison.** The same step with $\alpha\equiv\tfrac12$ is a deterministic midpoint rule. Its noise variance $s^2(\tfrac12)=h\,p(a)$ is the same, but $1-c(\tfrac12)^2=a\,\big(p(a)-a^3/4\big)$, so its variance factor is $\rho_{1/2}(a)=p(a)/\big(p(a)-a^3/4\big)=1+a^3/8+O(a^4)$, and its stationary law is exactly Gaussian. On Gaussian targets, randomizing $\alpha$ adds $a^4/12$ to the mean-square multiplier, raises the leading covariance error from $a^3/8$ to $a^3/6$, and, as shown next, makes the stationary law non-Gaussian.

### RMP: the stationary law is a Gaussian mixture

**Step 5: conditional Gaussianity.** Conditional on the whole sequence $(\alpha_k)$, the recursion is linear and Gaussian with independent coordinates. Hence, given $(\alpha_k)$ and a start $x_0\sim\mathcal N\big(0,\mathrm{diag}(V_{0,1},\dots,V_{0,d})\big)$, we have $x_k\sim\mathcal N\big(0,\mathrm{diag}(V_{k,1},\dots,V_{k,d})\big)$ with
$$V_{k+1,i}=c_i(\alpha_k)^2\,V_{k,i}+s_i^2(\alpha_k),$$
where $s_i^2$ is $s^2$ with $a=a_i$. The same $\alpha_k$ enters every coordinate, so the conditional variances $(V_{k,i})_i$ are random and dependent across $i$. At stationarity, the law $\mu_h$ of the chain is the law of $\mathrm{diag}(V_1,\dots,V_d)^{1/2}Z$, where $Z\sim\mathcal N(0,I_d)$ is independent of $(V_1,\dots,V_d)$, and $(V_1,\dots,V_d)$ is drawn from the stationary law of the $V$-recursion (unique for $hL<a_\star$, since that recursion contracts in mean). So $\mu_h$ is a Gaussian scale mixture: its coordinates are uncorrelated but not independent, and $\mu_h\neq\gamma_{S_h}$.

**Step 6: exact fourth moments.** Assume $hL<1$. Then $0<c_i(\alpha)<1$ for all $\alpha\in[0,1]$, so all moments are finite. Write $C_i=c_i(\alpha)^2$ and $v_i=\sigma_i^2\rho(a_i)$. At stationarity the current $(V_i,V_j)$ is independent of the fresh $\alpha$, so taking expectations in $V_i'V_j'=(C_iV_i+s_i^2)(C_jV_j+s_j^2)$ gives
$$\mathbb E[V_iV_j]=\frac{\mathbb E[C_is_j^2]\,v_i+\mathbb E[s_i^2C_j]\,v_j+\mathbb E[s_i^2s_j^2]}{1-\mathbb E[C_iC_j]}.$$
Each expectation is the average of a polynomial in $\alpha$ of degree at most 4, computed with $\mathbb E\alpha^n=1/(n+1)$, so $\mathbb E[V_iV_j]$ is an explicit rational function of $(a_i,a_j)$ times $\sigma_i^2\sigma_j^2$. The fourth moments of $\mu_h$ follow: $\mathbb E[x_i^2x_j^2]=\mathbb E[V_iV_j]$ for $i\neq j$, and $\mathbb E[x_i^4]=3\,\mathbb E[V_i^2]$, whereas $\gamma_{S_h}$ would give $v_iv_j$ and $3v_i^2$. The normalized excess
$$R_{ij}=\frac{\mathrm{Cov}(V_i,V_j)}{v_iv_j}=\frac{\mathrm{Cov}\big(C_iv_i+s_i^2,\;C_jv_j+s_j^2\big)}{v_iv_j\,\big(1-\mathbb E[C_iC_j]\big)}=\frac{(a_ia_j)^2}{6\,(a_i+a_j)}\big(1+O(a_i+a_j)\big),$$
so that $R_{ii}=a_i^3/12+O(a_i^4)$, is an exact, closed-form certificate that $\mu_h$ is not Gaussian. (For the middle expression, subtract $v_iv_j=\mathbb E[C_iv_i+s_i^2]\,\mathbb E[C_jv_j+s_j^2]$ from both sides of the fixed-point equation and regroup: $\mathrm{Cov}(V_i,V_j)=\mathbb E[C_iC_j]\,\mathrm{Cov}(V_i,V_j)+\mathrm{Cov}(C_iv_i+s_i^2,\,C_jv_j+s_j^2)$.)

### RMP: what the exact moments give for KL and $W_2$

**Step 7: exact split of KL, and exact lower bounds.** Both $\log\pi$ and $\log\gamma_{S_h}$ are quadratic polynomials, and $\mu_h$ has mean $0$ and covariance $S_h$, so $\int\mu_h\log\pi=\int\gamma_{S_h}\log\pi$ and $\int\mu_h\log\gamma_{S_h}=\int\gamma_{S_h}\log\gamma_{S_h}$. Subtracting gives the exact split
$$\mathrm{KL}(\mu_h\,\|\,\pi)=\underbrace{\mathrm{KL}(\gamma_{S_h}\,\|\,\pi)}_{\text{covariance part}}+\underbrace{\mathrm{KL}(\mu_h\,\|\,\gamma_{S_h})}_{\text{shared-midpoint part}},\qquad \mathrm{KL}(\gamma_{S_h}\,\|\,\pi)=\frac12\sum_{i=1}^d\Big(\rho(a_i)-1-\log\rho(a_i)\Big)=\frac{h^6}{144}\sum_{i=1}^d\lambda_i^6\,\big(1+O(hL)\big),$$
using $\rho-1-\log\rho=\tfrac12(\rho-1)^2+O((\rho-1)^3)$ and $\rho(a)-1=a^3/6+O(a^4)$. For $W_2$, the Gelbrich bound [gelbrich1990] with commuting covariances gives
$$W_2^2(\mu_h,\pi)\ \ge\ W_2^2(\gamma_{S_h},\pi)=\sum_{i=1}^d\sigma_i^2\Big(\sqrt{\rho(a_i)}-1\Big)^2=\frac{h^6}{144}\sum_{i=1}^d\lambda_i^5\,\big(1+O(hL)\big).$$
Both covariance terms are exact closed forms in $h$ and the eigenvalues of $\Sigma$, and both are lower bounds on the true stationary bias. For an isotropic spectrum they grow like $d$ (KL) and $\sqrt d$ ($W_2$).

**Step 8: the shared-midpoint part, to second order.** The term $\mathrm{KL}(\mu_h\|\gamma_{S_h})$ has no closed form, but a second-order expansion is available. Write $V_i=v_i(1+u_i)$, let $u'$ be an independent copy of $u$, and let $\varphi_s$ be the $\mathcal N(0,s)$ density on $\mathbb R$. For scalars with $uu'<1$, $\int\varphi_{v(1+u)}\varphi_{v(1+u')}/\varphi_v\,dx=(1-uu')^{-1/2}$. Each $V_i$ lies between $s^2_{\min}/(1-c_{\min}^2)$ and $s^2_{\max}/(1-c_{\max}^2)$, the values for the extreme $c_i(\alpha)$ and $s_i^2(\alpha)$, so $u_i>-1$, and $u_i<1$ whenever $a_i<0.442$. Hence for $hL<0.442$ we have $u_iu_i'<1$ almost surely and the exact identity $1+\chi^2(\mu_h\|\gamma_{S_h})=\mathbb E\prod_i(1-u_iu_i')^{-1/2}$ holds. Expanding the logarithm of the product to second order in $u$, and using $\mathrm{KL}\approx\tfrac12\chi^2$ for small perturbations, gives
$$\mathrm{KL}(\mu_h\,\|\,\gamma_{S_h})\approx\frac1{16}\Big(2\sum_iR_{ii}^2+\sum_{i,j}R_{ij}^2\Big),$$
to leading order when $\sum_{i,j}R_{ij}^2$ is small. For an isotropic spectrum ($a_i=a$ for all $i$), every $R_{ij}$ equals the same $R\approx a^3/12$, so the shared-midpoint part is about $(d^2+2d)\,a^6/2304$, against $d\,a^6/144$ for the covariance part. Their ratio is about $(d+2)/16$, so the shared midpoint dominates once $d$ exceeds about $14$. Keeping the total below $\varepsilon^2$ (that is, $\sqrt{\mathrm{KL}}\le\varepsilon$) then requires $hL$ of order $(\varepsilon/d)^{1/3}$, so at $\kappa=1$ the number of steps grows like $(d/\varepsilon)^{1/3}$, up to logarithmic factors. This has the same $d^{1/3}\varepsilon^{-1/3}$ scaling as the Gaussian lower bound reported in [chen2026], which we take only as a consistency check on the algebra. The expansion is an approximation; checking it against the exact one-dimensional reduction available for isotropic spectra is the first planned measurement.

### Planned extensions

1. **Underdamped randomized midpoint** [shen2019]. Per eigen-direction the state is $z=(x,\text{velocity})\in\mathbb R^2$, and given $\alpha$ the step is $z\mapsto A_\alpha z+\zeta_\alpha$, with $\zeta_\alpha$ centered Gaussian of covariance $Q_\alpha$, both obtained from the exact Ornstein-Uhlenbeck integrals with the gradient frozen at the random midpoint. The stationary second moment solves $S=\mathbb E_\alpha[A_\alpha SA_\alpha^\top]+\mathbb E_\alpha[Q_\alpha]$, i.e. $\mathrm{vec}(S)=\big(I-\mathbb E_\alpha[A_\alpha\otimes A_\alpha]\big)^{-1}\mathrm{vec}\big(\mathbb E_\alpha[Q_\alpha]\big)$, with the average taken inside the Kronecker product. The fourth-moment recursion is analogous, with the scalar $V$ replaced by a $2\times2$ conditional covariance. We have not yet derived $A_\alpha$ and $Q_\alpha$ in closed form; the repository implements the generic solver and checks it on the overdamped case.
2. **Deterministic midpoint schemes**, such as those analyzed in [chen2026], have deterministic coefficients on Gaussian targets, so their stationary laws are Gaussian and their $W_2$ and KL bias are exact closed forms, as for ULA and the fixed-midpoint rule above.
3. **Non-Gaussian perturbation probe.** Replace $f$ by $f_\tau(x)=\tfrac12x^\top\Sigma^{-1}x+\tau\sum_ig(x^{(i)})$ with $g(t)=\log\cosh t$ and $\tau\ge0$, and measure by simulation how quickly the bias leaves the Gaussian predictions as $\tau$ grows. We make no prediction about the outcome.

## Methodology

**Modules.** (1) `src/exact_gaussian.py`: closed forms for ULA and for overdamped RMP (stationary and finite-$k$ covariance, the stability threshold $a_\star$, the exact fourth moments $\mathbb E[V_iV_j]$ and excess $R_{ij}$, the covariance parts of KL and $W_2$, and the second-order shared-midpoint KL, which is labeled as an approximation), plus a generic solver for $S=\mathbb E_\alpha[A_\alpha SA_\alpha^\top]+\mathbb E_\alpha[Q_\alpha]$. (2) `src/samplers.py`: simulation of ULA, MALA and overdamped RMP, with one $\alpha$ per step shared across coordinates, for Monte Carlo cross-checks. (3) `src/perturbations.py`: the non-Gaussian probe. (4) `src/report.py`: plotting.

**Correctness checks.** The unit tests compare the closed form $S_h=\Sigma\,\rho(h\Sigma^{-1})$ for a non-diagonal $\Sigma$ with the matrix fixed point $\mathrm{vec}(S)=\big(I-\mathbb E[C\otimes C]\big)^{-1}\mathrm{vec}\big(\mathbb E[Q]\big)$, where $C(\alpha)=I-h\Sigma^{-1}+\alpha h^2\Sigma^{-2}$ and $Q(\alpha)=2h\big(I-2\alpha h\Sigma^{-1}+\alpha h^2\Sigma^{-2}\big)$ are the matrix forms of $c(\alpha)$ and $s^2(\alpha)$, and with direct iteration of the covariance recursion. They also compare the exact second and fourth moments with Monte Carlo simulation of the chain, check $R_{ij}$ against an independent quadrature of the covariance formula, and check that the fixed choice $\alpha=\tfrac12$ gives the different value $\rho_{1/2}$. These tests check algebra, not research questions; any new scheme must pass the same checks before it is used as a reference.

**Spectra.** Precision spectra will be isotropic; geometric between $m$ and $L$ with $\kappa$ swept; and $k$ eigenvalues equal to $L$ with the rest equal to $m$, to separate the roles of $d$, $\kappa$ and the sums $\sum_i\lambda_i^p$ that appear above.

**Synthetic data statement.** All targets in the proposed repository are synthetic, analytically specified distributions (Gaussians and clearly labeled perturbations of them). No real datasets are used, and any generated samples are labeled as mock/synthetic.

**AI Assistance Disclosure.** AI tools (Claude) were used for literature synthesis, ideation, and draft preparation of this proposal, including the derivations and the code. An AI review checked the draft before release, and the authors take responsibility for all of it, including the derivations and citations. Every derivation should be checked independently, and we welcome corrections.

## Planned Evaluation & Baselines

**Protocol only. No results have been obtained.**

**Baselines.** ULA (exact, known); overdamped RMP (exact second and fourth moments, derived above); the fixed-midpoint rule $\alpha\equiv\tfrac12$ (exact Gaussian law); underdamped RMP [shen2019] (planned); deterministic midpoint schemes as in [chen2026] (planned); and MALA [chewi2021] as a simulation-only reference, which the exact module does not cover because its accept/reject step is nonlinear.

**Planned measurements.**
1. Check the second-order shared-midpoint KL against the exact one-dimensional reduction for isotropic spectra. Both $\mu_h$ and $\pi$ are then rotation invariant, so KL and $W_2$ reduce to the law of $|x|^2=V\,|Z|^2$, a scale mixture of $\chi^2_d$ laws, which can be evaluated by one-dimensional quadrature over a grid of $(a,d)$.
2. For each spectrum family and method, compute the covariance and shared-midpoint parts of the stationary bias as functions of $h$, $d$ and $\kappa$, and the step size $h^*(\varepsilon,d,\kappa)$ needed to bring the bias below $\varepsilon$; fit exponents in $d$ and $\kappa$.
3. Compare with published bounds, noting that the metrics differ: [shen2019] and [yu2023] state $W_2$ bounds, [altschuler2026] KL bounds, and [chen2026] state their deterministic-method bounds in $W_2$. Any numbers will be quoted from those papers and attributed. Gaussian targets are a special case: they can show how large a bound must be at least, but cannot confirm that a worst-case bound is tight.
4. For the perturbation probe, measure the Monte Carlo bias against the Gaussian prediction as $\tau$ grows, with confidence intervals from independent replicates.

**Sanity criteria.** Closed forms equal the matrix fixed point to numerical precision; Monte Carlo agrees with the exact moments within sampling error; and the thresholds $hL<2$ (ULA) and $hL<a_\star$ (RMP) are reproduced.

## Limitations & Open Questions

- **Gaussians are special.** On a Gaussian target the third derivative of $f$ vanishes, so error terms that drive worst-case bounds may be absent. A Gaussian calculation can show that a bound must be at least as large as the Gaussian bias, never that it is tight in general.
- **Second order only for the mixture part.** The shared-midpoint KL above is an expansion that is valid when $\sum_{i,j}R_{ij}^2$ is small. Outside that regime, and for non-isotropic spectra, only the exact moments and simulation are available.
- **Stationary bias versus finite time.** Finite-$k$ moments are also exact, but the derivations above focus on stationarity. The lower bound in [chen2026] concerns a benign initialization, so the finite-$k$ version is the closer comparison.
- **Step-size range.** For $1\le hL<a_\star$, some $c(\alpha)$ exceed $1$ and higher moments of the stationary law can diverge, so the fourth-moment formulas are stated for $hL<1$.
- **Underdamped coefficients not yet derived.** That extension is a plan, and errors there are possible.
- **Open questions for experts.** Which parts of this calculation are already standard, for example in the Gaussian calculation of [chen2026], so that the testbed can cite rather than re-derive them? Since Gaussian targets appear to force only $d^{1/3}\varepsilon^{-1/3}$ steps at bounded $\kappa$, while the KL guarantee in [altschuler2026] is $\tilde O(\sqrt d/\varepsilon)$, which non-Gaussian features would be most informative to probe? Which spectra or metrics would researchers working on midpoint methods find most useful?

## References

- [chen2026] Chen, F., Chewi, S., Lu, J., Zhang, M. S. *Sharp dimensional analysis of midpoint methods for Langevin sampling*. arXiv preprint arXiv:2610.06308, 2026. https://arxiv.org/abs/2610.06308
- [altschuler2026] Altschuler, J. M., Chewi, S. *Shifted Composition III: Local Error Framework for KL Divergence*. Foundations of Computational Mathematics 26, 3331-3391, 2026. https://doi.org/10.1007/s10208-026-09753-x (arXiv:2412.17997)
- [shen2019] Shen, R., Lee, Y. T. *The Randomized Midpoint Method for Log-Concave Sampling*. NeurIPS 2019. https://arxiv.org/abs/1909.05503
- [he2020] He, Y., Balasubramanian, K., Erdogdu, M. A. *On the Ergodicity, Bias and Asymptotic Normality of Randomized Midpoint Sampling Method*. NeurIPS 2020. https://arxiv.org/abs/2011.03176
- [yu2023] Yu, L., Karagulyan, A., Dalalyan, A. *Langevin Monte Carlo for strongly log-concave distributions: Randomized midpoint revisited*. arXiv preprint arXiv:2306.08494, 2023. https://arxiv.org/abs/2306.08494
- [wibisono2018] Wibisono, A. *Sampling as optimization in the space of measures: The Langevin dynamics as a composite optimization problem*. Conference on Learning Theory (COLT) 2018. https://arxiv.org/abs/1802.08089
- [durmus2017] Durmus, A., Moulines, E. *Nonasymptotic convergence analysis for the unadjusted Langevin algorithm*. Annals of Applied Probability 27(3), 2017. https://arxiv.org/abs/1507.05021
- [dalalyan2017] Dalalyan, A. S. *Theoretical guarantees for approximate sampling from smooth and log-concave densities*. Journal of the Royal Statistical Society: Series B 79(3), 2017. https://arxiv.org/abs/1412.7392
- [gelbrich1990] Gelbrich, M. *On a formula for the L2 Wasserstein metric between measures on Euclidean and Hilbert spaces*. Mathematische Nachrichten 147, 185-203, 1990. https://doi.org/10.1002/mana.19901470121
- [chewi2021] Chewi, S., Lu, C., Ahn, K., Cheng, X., Le Gouic, T., Rigollet, P. *Optimal dimension dependence of the Metropolis-Adjusted Langevin Algorithm*. Conference on Learning Theory (COLT) 2021. https://arxiv.org/abs/2012.12810
