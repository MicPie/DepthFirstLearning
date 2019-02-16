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
##### :black_small_square: 1.31 Consider two variables x and y having joint distribution p(x, y). Show that the differential entropy of this pair of variables satisfies H(x,y) <= H(x) + H(y) with equality if, and only if, x and y are statistically independent.
H(x,y) = H(x) + H(y) if H(x|y) = H(x) and H(y|x) = H(y)\
H(x) + H(y|x) = H(y) + H(x|y) = H(x) + H(y)

No mutual information if x and y are independent:\
I(x,y) = H(x) + H(y) - H(x,y) because H(x,y) = H(x) + H(y)


##### :black_small_square: 1.36 A strictly convex function is defined as one for which every chord lies above the function. Show that this is equivalent to the condition that the second derivative of the function be positive.
Lookup 2nd derivative


##### :black_small_square: 1.37 Using the definition (1.111) together with the product rule of probability, prove the result (1.112).
(1.112) H[x,y] = H[y|x] + H[x]


##### :black_small_square: 1.38 Using proof by induction, show that the inequality (1.114) for convex functions implies the result (1.115).
???


##### :black_small_square: 1.39 Consider two binary variables x and y having the joint distribution given in Table 1.3.
The joint distribution p(x, y) for two binary variables x and y used:\

| x/y | 0 | 1 | p(x) |
| --- | --- | --- | --- |
| 0 | 1/3 | 1/3 | 2/3 |
| 1 | 0 | 1/3 | 1/3 |
| p(y) | 1/3 | 2/3 | 1 |

H(x) =  ∑(over x) p(x) log2(1/p(x)) = - ∑(over x) p(x) log2(p(x))\
H(x|y) =  - ∑(over x,y) p(x,y) log2(p(x,y)/p(y))\
H(x,y) =  - ∑(over x,y) p(x,y) log2(p(x,y))


https://en.wikipedia.org/wiki/Joint_probability_distribution#Draws_from_an_urn


**CHECK PRML Page 14?**

Evaluate the following quantities:

(a) H[x]

H(x) = - 1/3 * log2(1/3) - 0 * log2(0) ??? (very likely wrong?)\
H(x) = - 2/3 * log2(2/3) - 1/3 * log2(1/3) = 


(b) H[y]

H(y) = - 1/3 * log2(1/3) - 1/3 * log2(1/3) ??? (very likely wrong?)\
H(x) = - 1/3 * log2(1/3) - 2/3 * log2(2/3) = 


(c) H[y|x]

H(y|x) = 


(d) H[x|y]

H(x|y) = 


(e) H[x, y]

H(x,y) = 


(f) I[x, y]

I(x,y) = 


##### :black_small_square: 1.41 Using the sum and product rules of probability, show that the mutual information I(x, y) satisfies the relation (1.121).
(1.121) I[x, y] = H[x] − H[x|y] = H[y] − H[y|x]


#### :black_medium_small_square: How is Mutual Information similar to correlation? How are they different? Are they directly related under some conditions?
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

https://en.wikipedia.org/wiki/Entropy_(information_theory)#Further_properties --> Jensen inequality?


Very nice picture for p(x,y) https://en.wikipedia.org/wiki/Joint_probability_distribution

#### :black_medium_small_square: In classification problems, minimizing cross-entropy loss is the same as minimizing the KL divergence of the predicted class distribution from the true class distribution. Why do we minimize the KL, rather than other measures, such as L2 distance?
See solution at https://www.depthfirstlearning.com/2018/InfoGAN#1-information-theory.


## 2 Generative Adversarial Networks (GAN)

### Required Reading:
 - [ ] [JS Divergence](https://en.wikipedia.org/wiki/Jensen–Shannon_divergence)
 - [ ] [The original GAN paper](https://arxiv.org/abs/1406.2661)
