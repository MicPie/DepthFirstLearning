# InfoGAN
Material from the lecture about InfoGAN from https://www.depthfirstlearning.com/2018/InfoGAN.

## 1 Information Theory

### Required Reading:
- [x] Chapter 1.6 from [Pattern Recognition and Machine Learning / Bishop.](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf) (“PRML”)
- [x] [A good intuitive explanation of Entropy, from Quora.](https://www.quora.com/What-is-an-intuitive-explanation-of-the-concept-of-entropy-in-information-theory/answer/Peter-Gribble)

### Optional Reading:
- [x] [Notes on Kullback-Leibler Divergence and Likelihood Theory](https://arxiv.org/pdf/1404.2000.pdf)
- [x] [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (Highly recommended!)<br>

For more perspectives and deeper dependencies, see Metacademy:
- [ ] [Entropy](https://metacademy.org/graphs/concepts/entropy)
- [ ] [Mutual Information](https://metacademy.org/graphs/concepts/mutual_information)
- [ ] [KL diverence](https://metacademy.org/graphs/concepts/kl_divergence)

### Questions:
#### :black_medium_small_square:From PRML:
##### :white_small_square: 1.31 Consider two variables x and y having joint distribution p(x, y). Show that the differential entropy of this pair of variables satisfies H(x,y) <= H(x) + H(y) with equality if, and only if, x and y are statistically independent.
if H(x|y) = H(x) and H(y|x) = H(y)\
H(x) + H(y|x) = H(y) + H(x|y) = H(x) + H(y)\
H(x,y) = H(x) + H(y)


No mutual information if x and y are independent:\
I(x,y) = H(x) + H(y) - H(x,y) because H(x,y) = H(x) + H(y)


##### :black_small_square: 1.36 A strictly convex function is defined as one for which every chord lies above the function. Show that this is equivalent to the condition that the second derivative of the function be positive.
Lookup 2nd derivative


##### :black_small_square: 1.37 Using the definition (1.111) together with the product rule of probability, prove the result (1.112).
(1.111) H[y|x] = − ∫∫ p(y, x) lnp(y|x) dy dx\
(1.112) H[x,y] = H[y|x] + H[x]


##### :black_small_square: 1.38 Using proof by induction, show that the inequality (1.114) for convex functions implies the result (1.115).
(1.114) f(λa + (1 − λ)b)  λf(a) + (1 − λ)f(b)\
(1.115) (Insert picture here)


##### :white_small_square: 1.39 Consider two binary variables x and y having the joint distribution given in Table 1.3.
The joint distribution p(x, y) for two binary variables x and y used:

| y/x | x0 | x1 | p(y) |
| --- | --- | --- | --- |
| y0 | 1/3 | 0 | 1/3 |
| y1 | 1/3 | 1/3 | 2/3 |
| p(x) | 2/3 | 1/3 | 1 |

H(x) =  ∑(over x) p(x) log2(1/p(x)) = - ∑(over x) p(x) log2(p(x))\
H(x|y) =  - ∑(over x,y) p(x,y) log2(p(x,y)/p(y))\
H(x,y) =  - ∑(over x,y) p(x,y) log2(p(x,y))


Evaluate the following quantities:

See PRML p13.

(a) H[x]

p(x0) = p(x0,y0) + p(x0,y1) = 1/3 + 1/3 = 2/3\
p(x1) = p(x1,y0) + p(x1,y1) = 0 + 1/3 = 1/3\
H(x) = H(x0) + H(x1) = - 2/3 log2(2/3) - 1/3 log(1/3) = 0,918 bits 


(b) H[y]

p(y0) = p(x0,y0) + p(x1,y0) = 1/3 + 0 = 1/3\
p(y1) = p(x0,y1) + p(x1,y1) = 1/3 + 1/3 = 2/3\
H(y) = H(y0) + H(y1) = - 1/3 log2(1/3) - 2/3 log(2/3) = 0,918 bits


(c) H[y|x]

p(X = xi) = Sum over j to L (p(X=xi,Y=yj))\
p(Y=yj|X=xi) nij / ci  (with ci = sum over j for nij)\
p(y0|x0) = (1/3) / (1/3 + 1/3) = 1/2\
p(y1|x0) = (1/3) / (1/3 + 1/3) = 1/2\
p(y0|x1) = (0) / (1/3) = 0\
p(y1|x1) = (1/3) / (1/3) = 1\
H(Y|X) = sum over x,y( -(px,y) log2(p(y|x)) (formula from https://colah.github.io/posts/2015-09-Visual-Information/) \
H(y|x) = 0,667 bits



(d) H[x|y]

p(x0|y0) = (1/3) / (1/3 + 0) = 1\
p(x1|y0) = (0) / (1/3 + 0) = 0\
p(x0|y1) = (1/3) / (1/3 + 1/3) = 1/2\
p(x1|y1) = (1/3) / (1/3 + 1/3) = 1/2\
H(X|Y) = sum over x,y( -(px,y) log2(p(x|y))\
H(x|y) = 0,667 bits


(e) H[x, y]

H(x,y) = H(y) + H(x|y) = 0,918 + 0,667 bits = 1,585 bits


(f) I[x, y]

I(x,y) = H(x) + H(y) - H(x,y) = H(x) − H(x|y) = H(y) − H(y|x) = 0,252 bits

Other resources:
https://faculty.math.illinois.edu/~hildebr/408/408jointdiscrete.pdf
http://users.stat.ufl.edu/~abhisheksaha/sta4321/lect27.pdf
http://homepage.stat.uiowa.edu/~rdecook/stat2020/notes/ch5_pt1.pdf
https://www.khanacademy.org/math/ap-statistics/probability-ap/stats-conditional-probability/a/conditional-probability-using-two-way-tables


##### :black_small_square: 1.41 Using the sum and product rules of probability, show that the mutual information I(x, y) satisfies the relation (1.121).
(1.121) I[x, y] = H[x] − H[x|y] = H[y] − H[y|x]

I(x,y) = H(x) + H(y) - H(x,y)


#### :white_medium_small_square: How is Mutual Information similar to correlation? How are they different? Are they directly related under some conditions?
“Intuitively, mutual information measures the information that X and Y share: It measures how much knowing one of these variables reduces uncertainty about the other. For example, if X and Y are independent, then knowing X does not give any information about Y and vice versa, so their mutual information is zero. At the other extreme, if X is a deterministic function of Y and Y is a deterministic function of X then all information conveyed by X is shared with Y: knowing X determines the value of Y and vice versa. As a result, in this case the mutual information is the same as the uncertainty contained in Y (or X) alone, namely the entropy of Y (or X). Moreover, this mutual information is the same as the entropy of X and as the entropy of Y. (A very special case of this is when X and Y are the same random variable.)

Mutual information is a measure of the inherent dependence expressed in the joint distribution of X and Y relative to the joint distribution of X and Y under the assumption of independence. Mutual information therefore measures dependence in the following sense: I(X;Y) = 0 if and only if X and Y are independent random variables. This is easy to see in one direction: if X and Y are independent, then p(x,y) = p(x) p(y), and therefore:

Log(p(x,y) / (p(x) p(y)) = log(1) = 0. Moreover, mutual information is nonnegative (i.e. I(X;Y) >= 0 see below) and symmetric (i.e. I(X;Y) = I(Y;X) see below).”

“Not limited to real-valued random variables like the correlation coefficient, MI is more general and determines how similar the joint distribution p(x,y) is to the products of factored marginal distribution p(x) p(y). MI is the expected value of the pointwise mutual information (PMI).“ (https://en.wikipedia.org/wiki/Mutual_information)

Formulas see https://stats.stackexchange.com/questions/81659/mutual-information-versus-correlation

“Compare the two: each contains a point-wise "measure" of "the distance of the two rv's from independence" as it is expressed by the distance of the joint pmf from the product of the marginal pmf's: the Cov(X,Y) has it as difference of levels, while I(X,Y) has it as difference of logarithms.

And what do these measures do? In Cov(X,Y) they create a weighted sum of the product of the two random variables. In I(X,Y) they create a weighted sum of their joint probabilities.

So with Cov(X,Y) we look at what non-independence does to their product, while in I(X,Y) we look at what non-independence does to their joint probability distribution.

Reversely, I(X,Y) is the average value of the logarithmic measure of distance from independence, while Cov(X,Y) is the weighted value of the levels-measure of distance from independence, weighted by the product of the two rv's.

So the two are not antagonistic—they are complementary, describing different aspects of the association between two random variables. One could comment that Mutual Information "is not concerned" whether the association is linear or not, while Covariance may be zero and the variables may still be stochastically dependent. On the other hand, Covariance can be calculated directly from a data sample without the need to actually know the probability distributions involved (since it is an expression involving moments of the distribution), while Mutual Information requires knowledge of the distributions, whose estimation, if unknown, is a much more delicate and uncertain work compared to the estimation of Covariance.” 
(https://stats.stackexchange.com/questions/81659/mutual-information-versus-correlation)

Picture from https://www.quora.com/What-is-the-difference-between-mutual-information-and-correlation


#### :white_medium_small_square: In classification problems, minimizing cross-entropy loss is the same as minimizing the KL divergence of the predicted class distribution from the true class distribution. Why do we minimize the KL, rather than other measures, such as L2 distance?
See solution at https://www.depthfirstlearning.com/2018/InfoGAN#1-information-theory.

## 2 Generative Adversarial Networks (GAN)

### Required Reading:
- [x] [JS Divergence](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
- [x] [The original GAN paper](https://arxiv.org/abs/1406.2661)

https://en.wikipedia.org/wiki/Entropy_(information_theory)#Further_properties --> Jensen inequality?

### Questions:

##### :black_small_square: Prove that 0≤JSD(P||Q)≤1 bit for all P, Q. When are the bounds achieved?
https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

JSD(P||Q) = 1/2 * KL(P||M) + 1/2 * KL(Q||M)

M = 1/2 * (P+Q)

I(X;Z) = JSD(P||Q)

https://en.wikipedia.org/wiki/Jensen–Shannon_divergence#Bounds

“It follows from the above result that the Jensen–Shannon divergence is bounded by 0 and 1 because mutual information is non-negative and bounded by H(Z)=1. The JSD is not always bounded by 0 and 1: the upper limit of 1 arises here because we are considering the specific case involving the binary variable Z.” (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence#Relation_to_mutual_information)

https://dit.readthedocs.io/en/latest/measures/divergences/jensen_shannon_divergence.html#derivation


##### :white_small_square: What are the bounds for KL divergence? When are those bounds achieved?
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

D(p||q) = KL(p,q) = Σ p(x) log(p(x)/q(x))

0 ≤ KL ≤ ∞ (infinity)

KL(p,q) = 0:\
When the two probability distributions p,q are identical:\
p(x)/q(x) = 1 and log(p(x)/q(x)) = 0 which results in KL(p,q) = 0

KL(p,q) = ∞:\
“In the context of coding theory, DKL(P||Q) can be construed as measuring the expected number of extra bits required to code samples from P using a code optimized for Q rather than the code optimized for P.” (https://stats.stackexchange.com/questions/323069/can-kl-divergence-ever-be-greater-than-1)

When they are totally different you would need an infinite number of extra bits and, therefore, the cross entropy and the KL would be infinite.\
Hp(q) = -Σ q(x) log(p(x)) = ∞\
KL(p,q) = Hp(q) - H(q) = ∞

**Other and very likely better explanation:**<br>
p(x)/q(x) = ∞ and log(p(x)/q(x)) = ∞ which results in KL(p,q) = ∞<br>
https://stats.stackexchange.com/questions/351947/whats-the-maximum-value-of-kullback-leibler-kl-divergence<br>
https://en.wikipedia.org/wiki/Support_(mathematics)#In_probability_and_measure_theory<br>
https://math.stackexchange.com/questions/2859284/how-to-use-kullback-leibler-divergence-if-probability-distributions-have-differe<br>
https://arxiv.org/pdf/math/0209021.pdf<br>
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Motivation<br>
When the support of Q is not included in the support of P, i.e., Q has zero values where P is not zero, the upper bound of the KL is infinity.


##### :white_small_square: In the paper, why do they say “In practice, equation 1 may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence because they are clearly different from the training data. In this case, log(1−D(G(z)))saturates”?

See the derivative of the loss functions at x = 0 in the pictures at the next question. There the non-saturating loss (green) has a much higher derivative when the G is learning to fake the pictures better (-log(D(G(z)) = 0 or slightly higher) compared to the saturating loss function (blue).

Saturating loss = ln(1-x), derivation = -1/(1-x)\
Non-saturating loss = -ln(x), derivation = -1/x (= negative log-likelhood)

[WolframAlpha plot incl. derivations](https://www.wolframalpha.com/input/?i=plot+(ln(1-x))+and+(-ln(x))+and+derivation(ln(1-x))+and+derivation(-ln(x))for+x+from+0+to+1)\
![GAN_saturation vs non-saturating_G loss_figure 2](https://raw.githubusercontent.com/MicPie/DepthFirstLearning/master/InfoGAN/sat_vs_nonsat_g_loss.png)\
Plot from [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160) p.26 resembles the above plots.

How does this relate to this explanation: https://www.depthfirstlearning.com/assets/gan_gradient.pdf \
[WolframAlpha plot other explanation](https://www.wolframalpha.com/input/?i=plot+log(1+%E2%88%92+sigmoid(x))+and+log(sigmoid(x))+for+x+from+-20+to+20)\
Another explanation for the same effect?


##### :white_small_square: Implement a Colab that trains a GAN for MNIST. Try both the saturating and non-saturating ~~discriminator~~ generator (?) loss.

The generator losses can be implemented in PyTorch with [torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss) and using y = 0 or 1 to switch between fake and real:

ln = −yn⋅log(σ(xn))-(1−yn)⋅log(1−σ(xn))

See [PyTorch DCGAN MNIST implementation notebook](https://nbviewer.jupyter.org/github/MicPie/DepthFirstLearning/blob/master/InfoGAN/DCGAN_MNIST_v5.ipynb) (based on the [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#discriminator)).

**[NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160):**\
D loss (BCE, p.21, formula 8):\
J(D) = −y⋅log(D(x))-(1−y)⋅log(1−D(G(z)))\
Training is split in training with real data (y = 1, no y = 0) and in training with fake data (y = 0).

Saturating G loss (minimax, zero-sum game, p.22):\
J(D) = -J(G)\
J(G) = y⋅log(D(x))+(1−y)⋅log(1−D(G(z)))\
if y = 0: J(G) = log(1−D(G(z)))

Non-saturating G loss (p.22):\
Heuristic: "Instead of flipping the sign on the discriminator's cost to obtain a cost for the generator, we flip the target used to construct the cross-entropy cost." (p.22):\
invert labels with y = 1\
J(G) = -log(D(G(z)))


https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
https://gombru.github.io/2018/05/23/cross_entropy_loss/


## 3 InfoGAN

### Required Reading

- [x] InfoGAN
- [x] A correction to a proof in the paper

### Optional Reading
- [x] A blog post explaining InfoGAN

### Questions

##### :white_small_square: How does one compute logQ(c|x) in practice? How does this answer change based on the choice of the type of random variables in c? (What is logQ(c|x) when c is a Gaussian centered at fθ(x)? What about when c is the output of a softmax?)

"In practice, we parametrize the auxiliary distribution Q as a neural network. In most experiments, Q and D share all convolutional layers and there is one final fully connected layer to output parameters for the conditional distribution Q(c|x), ... For categorical latent code cj, we use the natural choice of softmax nonlinearity to represent Q(cj|x). For continuous latent code cj, there are more options depending on what is the true posterior P(cj|x). In our experiments, we have found that simply treating Q(cj|x) as a factored Gaussian is sufficient." ([InfoGAN paper](https://arxiv.org/abs/1606.03657), chapter 6, p.4)

https://www.quora.com/What-is-meant-by-factored-distribution


##### :white_small_square: Which objective in the paper can actually be optimized with gradient-based algorithms? How? (An answer to this needs to refer to “the reparameterization trick”)

"We note that LI(G,Q) is easy to approximate with Monte Carlo simulation. In particular, LI can be maximized w.r.t. Q directly and w.r.t. G via the reparametrization trick. Hence LI(G,Q) can be added to GAN’s objectives with no change to GAN’s training procedure and we call the resulting algorithm Information Maximizing Generative Adversarial Networks (InfoGAN)." ([InfoGAN paper](https://arxiv.org/abs/1606.03657), chapter 6, p.4)


##### :white_small_square: Why is an auxiliary Q distribution necessary?

"In practice, the mutual information term I(c;G(z,c)) is hard to maximize directly as it requires access to the posterior P(c|x). Fortunately we can obtain a lower bound of it by defining an auxiliary distribution Q(c|x) to approximate P(c|x):" ([InfoGAN paper](https://arxiv.org/abs/1606.03657), chapter 5, p.3)


##### :white_small_square: Draw a neural network diagram for InfoGAN

https://towardsdatascience.com/infogan-generative-adversarial-networks-part-iii-380c0c6712cd

##### :white_small_square: In the paper they say “However, in this paper we opt for simplicity by fixing the latent code distribution and we will treat H(c) as a constant.”. What if you want to learn the latent code (say, if you don’t know that classes are balanced in the dataset). Can you still optimize for this with gradient-based algorithms? Can you implement this on an intentionally class-imbalanced variant of MNIST?

It seems to be possible to optimize a probability density function with gradient-based algorithms:
- https://www.reddit.com/r/MachineLearning/comments/5us720/d_probability_density_estimation_using_neural/
- https://arxiv.org/pdf/1612.01474.pdf

Does the probability density function need to be based on the entire training data? Yes!?\
Would a moving probability density estimation make sense?


##### :white_small_square: In the paper they say “the lower bound … is quickly maximized to … and maximal mutual information is achieved”. How do they know this is the maximal value?

If the lower bound LI(G,Q) is maximized for MNIST with c ∼ Cat(K = 10, p = 0.1) the entropy H(c) can be calculated:\
H(c) = 10 * -1/10 * log(1/10) = ~2.30


##### :white_small_square: Open-ended question: Is InfoGAN guaranteed to find disentangled representations? How would you tell if a representation is disentangled?

You have a disentangled representation if the lower bound LI(G,Q) is maximized to H(c). However, to know the exact value of H(c) you would need to know the probability distribution (e.g., in case of classes the probability density function), which is not a straightforward task.


## Implementations
- Original implementation https://github.com/openai/InfoGAN
- https://github.com/Natsu6767/InfoGAN-PyTorch (most up-to-date implementation; why does the [DHead](https://github.com/Natsu6767/InfoGAN-PyTorch/blob/4586919f2821b9b2e4aeff8a07c5003a5905c7f9/models/mnist_model.py#L52) uses sigmoid and not tanh?)
- https://github.com/pianomania/infoGAN-pytorch
- https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py
- https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference, http://saxarona.github.io/mathjax-viewer/, https://arachnoid.com/latex/

## Other links
- https://github.com/zhangqianhui/AdversarialNetsPapers
- https://github.com/nightrome/really-awesome-gan
