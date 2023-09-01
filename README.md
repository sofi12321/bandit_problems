# Bandit problem solutions
This code contains epsilon-Greedy and Thompson Sampling algorithms for Bernoulli bandit problem.
Also, it solves Gaussian Bandit problem by Upper-Confidence Bound and Thompson Sampling algorithms.

As a simulation, a task for Correlated Travel Times was solved using Thomson Sampling algorithm. 



## Bandit's problem
This task is aimed at implementing algorithms to solve bandit's problems. To begin with, let's introduce this problem. Imagine that there is a machine with many arms, each of them has a probability of winning. A player comes in. He wants to get as many wins as possible. The intuitive solution is to hold the arm with the highest success probability. However, the player does not know these probabilities. In this case, he can follow different strategies, some of which are implemented in this solution (epsilon-Greedy, Thompson Sampling). Note that this is a bandit problem with Bernoulli's reward: the player pulls a bandit's arm and with some probability wins (receives +1 reward) or loses (receives +0 reward). 

Now suppose that each arm produces a reward as a real value from a normal distribution. My solution assumes that the arms have different mean values, but the same variance equal to 1. The player knows variance, but does not know means and wants to maximize his reward. This is the Bandit problem with Gaussian rewards, for which I implemented UCB and Thompson Sampling algorithms. 

### General solution
While writing solutions to these problems, I noticed that all the algorithms have the same structure. Before the game starts, the parameters of prior distribution are set. Then, at each time period, the attractiveness of each action is evaluated and one of them is chosen based on this. After that, the action is performed and some reward is awarded for it. The posterior distribution is calculated based on this and the prior is updated for the next time period. Additionally, the regret can be counted as the received reward minus the best one that could be obtained at that moment.

So, I decided to use OOP to solve the problem elegantly. The structure of my solution is the following: 
* GeneralBanditAlgorithm is a class that implements the main function - the solution as a sequence of steps discussed earlier. Each step is a function call that can be rewritten by child classes. The function that displays the information is also realized here. Note that all other algorithms are its childs, and them all has original_param1 and original_param2 as data from the real world, estimated_param1 and estimated_param2 as parameters for prior distribution. 
* BernoulliBandit and GaussianBandit are child classes of GeneralBanditAlgorithm. They implement getting a reward and updating the prior distribution (including calculating the posterior). Note that for both Bandits the prior and posterior are conjugate distributions. 
* BernoulliEGreedy, BernoulliTS, GaussianUCB and GaussianTS are children of one of the previous classes. They modify the estimation of the reward by the prior distribution. In addition, epsilon-Greedy rewrites the choice of action. 
* CorrelatedTravelTimeSolution is a child class of GeneralBanditAlgorithm, which implements the CorrelatedTravelTime problem from the tutorial. 


Now, let's overview each algorithm. 

### Epsilon Greedy Algorithm for Bernoulli reward
In this algorithm the prior distribution is set as beta distribution with parameters estimated_param1=alpha=1, estimated_param2=beta=1 for each arm. The real probabilities of each arm are stored in original_param1 and set as a random number from 0 to 1. 

The estimated success probability (theta) of each arm is the expectation of the prior distribution: $alpha/(alpha+beta)$. Every time-period with probability epsilon the player polls a random arm with probability proportional to the expected reward, otherwise he takes the action with the highest theta.
Regret counts as the difference between the real winning probability of the chosen arm and the highest possible success probability.

Note that the real reward is 1 if the chosen arm wins, or 0 otherwise. Victory is determined whether the random number is less than the real arm's success probability.
At each time period,  only the selected arm's posterior is calculated and its prior for the next step is updated: $alpha=alpha+reward$, $beta=beta+1-reward$. 

### Thompson Sampling Algorithm for Bernoulli reward
This algorithm differs from the epsilon greedy algorithm in not using epsilon. Theta (estimated success probability) is taken as a sample from the prior distribution (with parameters estimated_param1 as alpha & estimated_param2 as beta), and the action is chosen as the arm with the highest theta.

I ran the algorithms for the Bernoulli reward for 1000 time periods 10000 times and got the following graph. From it one can notice that Thompson's algorithm converges the best.

![Per-period regret for E-Greedy and Thompson Sampling for Binomial reward](https://github.com/sofi12321/bandit_problems/assets/95312480/1d613af7-706e-4702-a975-d175f42e0241)

### Upper-Confidence Bound Algorithm for Gaussian reward
In this algorithm the real reward (likelihood) for each arm is taken from normal distribution with mean=original_param1 and variance=original_param2=1. Note that mean is taken as random number from 1 to 10, the player does not know it. However, he knows that real variance are all equal to 1. 
Before the start the player set the prior distribution as normal with parameters estimated_param1=mu=0, estimated_param2=sigma^2=1 for each arm. 

The expected reward (theta) of each arm is equal to the exploitation_part + confidence_factor*exploration_part, where exploitation_part is the expectation of the prior distribution; confidence_factor is a constant during algorithm initialization, which represents how interested we are in exploring arms; exploration_part = $\sqrt{\frac{\ln{N+1}}{n+1}}$, n is a number of times the arm was pulled, N is the overall number of performed trials. Every time-period the arm with the highest theta is taken. 

The real reward is a sample from likelihood. Regret counts as the difference between the obtained reward and the maximal possible. 
The posterior is calculated and prior is updated as conjugate distributions:

$\mu_{posterior} = \frac{1}{\frac{1}{\sigma^2_{prior}} + \frac{1}{\sigma^2_{likelihood}}}(\frac{\mu_{prior}}{\sigma^2_{prior}} + \frac{reward}{\sigma^2_{likelihood}})$

$\sigma_{posterior} = \frac{1}{\frac{1}{\sigma^2_{prior}} + \frac{1}{\sigma^2_{likelihood}}}$

### Thompson Sampling Algorithm for Gaussian reward
This algorithm differs from the UCB algorithm in not using confidence factor, theta estimation and arm choosing. 
Theta (estimated success probability) is taken as a sample from the prior distribution (with parameters estimated_param1 as mean & estimated_param2 as variance), and the action is chosen as the arm with the highest theta.


I ran the algorithms for the Gaussian reward for 500 time periods 10000 times and got the following graph. From it one can notice that Thompson's algorithm converges the best.

![Per-period regret for UCB and Thompson Sampling for Gaussian reward](https://github.com/sofi12321/bandit_problems/assets/95312480/a8ecfaa9-bfd3-4863-ba7d-78f7d97f6984)


## Thompson Sampling Algorithm for Correlated Travel Time Problem
The Correlated Travel Time problem is the problem of finding the optimal path in the context of unknown travel time on each road. Imagine that a person travels from home (source) to work (destination) every day. The roads between them create a binomial bridge. The picture shows a binomial bridge for 6 stages (number of roads on the trip). Note that all edges between the origin and the middle are in the upper half, the rest are in the lower half.

![Example of the binomial bridge](https://github.com/sofi12321/bandit_problems/assets/95312480/84fef280-11aa-4e40-9d4b-1027cf532321)

The person wants to spend as less time on the road as possible, but does not know which route is faster. In reality, the travel time on each road at each day (time period) is a sample from lognormal distribution with parameters mu and sigma. Suppose in this problem the person does not know only the average travel time along each edge, and the sigma for all is 1.  The problem is complicated by the fact that the choice of one road affects the travel time on the other road, which is expressed in the covariance matrix, which the person also knows. 

Since I will be working mostly with edges, I represented one action as the choice of a road, and in one period of time the person chooses several edges. Therefore, the dimension of means is $E\times 1$, and for the covariance matrix it is $E\times E$, where E is the number of edges. Also, now the sample from the prior and likelihood distributions are taken as exponent in the power of sample from multivariate normal distribution of means and covariance matrix.

Moreover, the reward is multiplied by idiosyncratic_factor, time_factor and halves_factor taken from the lognormal distribution with parameters $(-1/6 \sigma^2, 1/3 \sigma^2)$. Idiosyncratic_factor represent some obstacles on each road, time_factor is common for all edges, for example, today's weather, halves_factor is the same for one half of the binomial bridge (one for upper half, another - for lower). 

Note that the task is to minimize the reward, or one can multiply reward by $-1$ and solve the maximization problem. When estimated rewards for each edge is calculated (using sample from prior distribution and parameters), the best path is chosen by Dijkstra's algorithm as one with the minimal path length. Then, the real rewards are obtained, and the reward for this time period is the sum of the edges rewards along the chosen path. The regret counts as the difference between the obtained reward and the minimal possible (calculated again with the Dijkstra's algorithm).
