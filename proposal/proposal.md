# Exact Gaussian Oracles for Diagnosing the Dimension Dependence of Discretized Langevin Samplers: An Early-Stage Testbed Proposal

**Aarush Kandukoori** (Carnegie Mellon University) — Lead Student Author  
**Aditya Kandukoori** (Poolesville High School) — Co-Author

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Recent theory on discretized Langevin samplers, including midpoint-type and high-accuracy methods, gives sharp dimension-dependent guarantees for log-concave targets. Checking such guarantees numerically is hard, because Monte Carlo error in the measurements can hide small discretization bias. We propose an early-stage, open-source testbed built on one observation: for Gaussian targets, one-step samplers with deterministic linear coefficients (such as the unadjusted Langevin algorithm, ULA) have a Gaussian stationary law whose covariance can be computed exactly, with no Monte Carlo noise. This turns asymptotic bias into a deterministic quantity that can be compared directly with theoretical bounds. For randomized-midpoint schemes the coefficients depend on a random point u, so the stationary law is generally a non-Gaussian mixture; only its stationary second moments follow from a fixed-point equation, and that extension is planned and not yet derived. We work out the ULA case in closed form and propose to extend the second-moment analysis to randomized-midpoint schemes. We then perturb the target in a controlled, non-Gaussian way to measure how far the Gaussian predictions carry over. This is a framework and plan only. It reports no experimental results, and all data will be clearly labeled synthetic. We are students seeking expert feedback on whether the testbed would be useful and on which diagnostics matter most. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, and the authors reviewed and take responsibility for all content.

## Introduction & Motivation

**Status.** This is an early-stage proposal by two students. It contains no experimental results, and we are seeking expert feedback before building anything substantial.

**Problem.** Theory for log-concave sampling gives non-asymptotic bounds on the step size and iteration count a discretized Langevin sampler needs to reach error $\varepsilon$ in dimension $d$ [dalalyan2017, durmus2017, cheng2018]. Randomized-midpoint discretizations are known to improve dimension dependence in the strongly log-concave setting [shen2019]. Numerically checking such statements is awkward. A Monte Carlo estimate of, say, the Wasserstein distance to the target has sampling error that grows with $d$ and can be larger than the discretization bias being measured.

**Idea.** For a Gaussian target $\pi=\mathcal N(0,\Sigma)$, the Langevin diffusion is an Ornstein-Uhlenbeck process. A one-step scheme with deterministic coefficients, such as ULA, is a linear Gaussian recursion, so its stationary law is Gaussian with a covariance that can be computed exactly. For randomized-midpoint schemes the coefficients depend on a random point $u$, so the stationary law is generally a non-Gaussian mixture; only its stationary second moments follow from the fixed-point equation, and this extension is planned and not yet derived. The asymptotic bias of ULA is then a deterministic function of the step size $h$, the spectrum of $\Sigma^{-1}$, and $d$. We propose a testbed that computes this function, plots it against theoretical bounds, and then measures how the predictions degrade under controlled non-Gaussian perturbations.

**Scope.** We do not propose a new sampler or a new bound. The contribution, to our knowledge, would be a small reproducible diagnostic tool. Whether it is useful is exactly what we would like feedback on.

## Related Work

- **Convergence of ULA.** Dalalyan, Durmus-Moulines, and Cheng-Bartlett give non-asymptotic guarantees for the unadjusted Langevin algorithm under smoothness and (strong) log-concavity. Durmus-Moulines measure error in total variation and Cheng-Bartlett in KL divergence [dalalyan2017, durmus2017, cheng2018].
- **Midpoint methods.** Shen and Lee introduce the randomized midpoint method, which is built on underdamped Langevin dynamics, for log-concave sampling and show improved complexity in the strongly log-concave setting [shen2019].
- **Metropolis-adjusted methods.** Chewi et al. study the optimal dimension dependence of MALA from a warm start [chewi2021].
- **Optimal transport view.** Langevin dynamics is the Wasserstein gradient flow of KL divergence [jko1998].
- **Diffusion models.** Score-based generative models run a discretized reverse-time SDE [song2021], and convergence of such samplers can be reduced to score-estimation error plus discretization error [chen2023]. The same exact-Gaussian idea may transfer to these samplers, which we list as an open question.

We have not exhaustively surveyed the literature and welcome pointers to existing exact-Gaussian diagnostics we may have missed.

## Proposed Framework

**Setup.** Let $\pi=\mathcal N(0,\Sigma)$ with $\Sigma=\mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)$ (WLOG after rotation, since the schemes below commute with orthogonal changes of basis). Define precisions $\lambda_i=\sigma_i^{-2}\in[m,L]$, so the potential $f(x)=\tfrac12 x^\top\Sigma^{-1}x$ is $m$-strongly convex and $L$-smooth.

**Worked case: ULA.** The update is $x_{k+1}=x_k-h\nabla f(x_k)+\sqrt{2h}\,\xi_k$, $\xi_k\sim\mathcal N(0,I)$. Coordinatewise, with $a_i=h\lambda_i$,
$$x^{(i)}_{k+1}=(1-a_i)\,x^{(i)}_k+\sqrt{2h}\,\xi^{(i)}_k.$$
This is stable iff $|1-a_i|<1$, i.e. $h<2/L$. The stationary variance $v_i$ solves $v_i=(1-a_i)^2v_i+2h$, so
$$v_i=\frac{2h}{1-(1-a_i)^2}=\frac{2h}{a_i(2-a_i)}=\frac{\sigma_i^2}{1-a_i/2}.$$
(The last step uses $h/a_i=\sigma_i^2$.) The stationary law is $\mathcal N(0,\mathrm{diag}(v_i))$. Both laws are centered Gaussians with commuting covariances, so the squared 2-Wasserstein distance to $\pi$ is
$$W_2^2=\sum_{i=1}^d\big(\sqrt{v_i}-\sigma_i\big)^2=\sum_{i=1}^d\sigma_i^2\Big((1-a_i/2)^{-1/2}-1\Big)^2.$$
For small $a_i$, $(1-a_i/2)^{-1/2}-1=a_i/4+O(a_i^2)$, hence
$$W_2^2=\frac{h^2}{16}\sum_{i=1}^d\lambda_i\,(1+O(h\lambda_i))\le\frac{h^2dL}{16}(1+O(hL)).$$
So on Gaussians, the asymptotic ULA bias in $W_2$ is of order $h\sqrt{\mathrm{tr}(\Sigma^{-1})}/4$, and reaching bias $\varepsilon$ needs $h\approx 4\varepsilon/\sqrt{\mathrm{tr}(\Sigma^{-1})}$. Since $m\le\lambda_i\le L$, we have $md\le\mathrm{tr}(\Sigma^{-1})\le Ld$, so this $h$ lies in $\big[4\varepsilon/\sqrt{dL},\,4\varepsilon/\sqrt{dm}\big]$. The smallest step, which is the worst case for cost, is $4\varepsilon/\sqrt{dL}$, reached when all $\lambda_i=L$. This is a checkable, noise-free reference curve.

**Extension to midpoint-type schemes (planned).** Shen and Lee's randomized midpoint method is built on underdamped Langevin dynamics [shen2019], so the per-coordinate state is (position, velocity), not a scalar. A step draws a random fraction $u\sim U[0,1]$ and uses a gradient evaluated at an intermediate point. On a Gaussian target, conditional on $u$ the step is still a linear Gaussian map $z\mapsto A_u z+\text{noise}_u$ on the 2-dimensional per-coordinate state $z$. Because the coefficients depend on the random $u$, the stationary law is generally a non-Gaussian mixture, not a Gaussian. Only its stationary second moments follow from the fixed-point equation, which is a $2\times2$ matrix Lyapunov-type equation $S=\mathbb E_u[A_u S A_u^\top]+\mathbb E_u[Q_u]$, where $Q_u$ is the covariance of $\text{noise}_u$, rather than a scalar one. We have **not** derived these coefficients here, and this extension is planned and not yet derived. The plan is to derive them symbolically (computer algebra) from the exact stochastic integrals of the underdamped OU process, and to cross-check them against large-sample simulation of the scheme as a unit test before using them as a reference.

**Non-Gaussian perturbation probe (planned).** Replace $f$ by $f_\epsilon(x)=\tfrac12 x^\top\Sigma^{-1}x+\epsilon\sum_i g(x^{(i)})$ for a smooth bounded-curvature $g$ (e.g. $\log\cosh$). For small $\epsilon$ we ask empirically how the measured bias departs from the Gaussian prediction. This tests where Gaussian intuition about step-size scaling stops being informative. We make no prediction about the outcome.

## Methodology

**Modules.** (1) An exact-Gaussian module that computes stationary covariances for each scheme as a function of $(h,d,\text{spectrum})$. Exact $W_2$/KL-to-$\pi$ formulas are used only for deterministic-coefficient schemes (e.g. ULA). For randomized schemes, whose stationary law is generally non-Gaussian, the module reports the exact covariance error (e.g. the Frobenius or relative error of the stationary covariance) and $W_2$/KL are estimated by simulation, or the law is derived separately. Gaussian closed forms are not applied to a non-Gaussian stationary law. (2) A simulation module implementing the same schemes for Monte Carlo cross-checks. (3) A perturbation module for the non-Gaussian probe. (4) A plotting and reporting module.

**Correctness checks.** The ULA closed form above will be checked against the simulation module in a unit test, with the pass criterion being agreement within Monte Carlo error at moderate $d$. The same test will gate every new scheme before it is used as a reference.

**Spectra.** Precision spectra will be: isotropic, geometric between $m$ and $L$ (condition number $\kappa=L/m$ swept), and a few-large-eigenvalues case, to separate dependence on $d$ from dependence on $\mathrm{tr}(\Sigma^{-1})$.

**Synthetic data statement.** All targets in the proposed repository are synthetic, analytically specified distributions (Gaussians and clearly labeled perturbations of them). No real datasets are used, and any generated samples are labeled as mock/synthetic.

**AI Assistance Disclosure.** AI tools (Claude) were used for literature synthesis, ideation, and draft preparation of this proposal. The authors reviewed the content and take responsibility for all of it, including the derivations and citations. Every derivation should be checked independently, and we welcome corrections.

## Planned Evaluation & Baselines

**Protocol only. No results have been obtained.**

**Baselines (to be implemented).** ULA; randomized midpoint [shen2019]; and, if feasible, MALA [chewi2021] as a Metropolis-adjusted reference in the simulation module (not covered by the exact-Gaussian module, since the accept/reject step is nonlinear).

**Planned measurements.**
1. For ULA and each spectrum, compute the exact asymptotic $W_2$ bias (and KL) as a function of $h$ and $d$. For randomized schemes, compute the exact stationary covariance error (e.g. Frobenius or relative error) and estimate $W_2$/KL by simulation, or derive the law separately. Fit the empirical exponent of the step size $h^*(\varepsilon,d)$ needed to reach a target bias $\varepsilon$.
2. Compare the fitted scaling to the relevant published bounds, noting that the metrics differ: [durmus2017] is stated in total variation, [cheng2018] in KL divergence, and [shen2019] gives improved complexity for a randomized midpoint method based on underdamped Langevin in the strongly log-concave setting. We do not currently cite a source for $W_2$ bounds and would add one whose abstract supports it. Any numbers will be quoted from those papers and attributed. We will state clearly that Gaussian targets are a special case and cannot confirm or refute worst-case bounds.
3. For the perturbation probe, measure Monte Carlo bias against the Gaussian prediction as $\epsilon$ grows, with confidence intervals from independent replicates.

**Sanity criteria.** Exact vs. simulated agreement within Monte Carlo error on the Gaussian case; stability threshold $h<2/L$ reproduced for ULA.

## Limitations & Open Questions

- **Gaussians are special.** Worst-case lower bounds in the literature are often attained by non-Gaussian or ill-conditioned constructions. Gaussian exactness gives a noise-free reference, not evidence about worst-case behavior.
- **Asymptotic bias only.** The stationary distribution does not capture mixing time or non-asymptotic error from a warm or cold start, though the same linear recursions would also give exact finite-$k$ laws, which we could add.
- **Midpoint coefficients not yet derived.** The extension in the Framework section is a plan, and errors there are possible.
- **Open questions for experts.** Which diagnostics would be most informative to researchers working on midpoint or high-accuracy samplers? Is there existing exact-Gaussian tooling we should reuse? Would the same idea be meaningful for accelerated or underdamped schemes, or for diffusion-model samplers [song2021, chen2023]?

## References

- [dalalyan2017] Dalalyan et al., *Theoretical guarantees for approximate sampling from smooth and log-concave densities*, Journal of the Royal Statistical Society: Series B 2017. https://arxiv.org/abs/1412.7392
- [durmus2017] Durmus et al., *Non-asymptotic convergence analysis for the Unadjusted Langevin Algorithm*, Annals of Applied Probability 2017. https://arxiv.org/abs/1507.05021
- [cheng2018] Cheng et al., *Convergence of Langevin MCMC in KL-divergence*, Algorithmic Learning Theory (ALT) 2018. https://arxiv.org/abs/1705.09048
- [shen2019] Shen et al., *The Randomized Midpoint Method for Log-Concave Sampling*, NeurIPS 2019. https://arxiv.org/abs/1909.05503
- [chewi2021] Chewi et al., *Optimal dimension dependence of the Metropolis-Adjusted Langevin Algorithm*, COLT 2021. https://arxiv.org/abs/2012.12810
- [jko1998] Jordan et al., *The Variational Formulation of the Fokker--Planck Equation*, SIAM Journal on Mathematical Analysis 1998. https://doi.org/10.1137/S0036141096303359
- [song2021] Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021. https://arxiv.org/abs/2011.13456
- [chen2023] Chen et al., *Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions*, ICLR 2023. https://arxiv.org/abs/2209.11215
