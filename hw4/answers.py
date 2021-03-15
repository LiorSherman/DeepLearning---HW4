r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=0.5,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2
    )

    # ========================
    return hp


def part1_aac_hyperparams():
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp = dict(
        batch_size=4,
        gamma=0.97,
        beta=1,
        delta=1,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # ========================
    return hp


part1_q1 = r"""
Subtracting a baseline helps to reduce the variance of the policy-gradient since the rewards are compared to some number
and their absolute value is not the only thing that matters. We can take two different trajectories from some batch
where t1 has positive total reward and t2 has negative total reward. In the vanilla case without baseline t1 will get
much higher probability then t2, but if we'll subtract a baseline the reward for t1 can be much smaller in even negative
so in this case the gradient will be different and the difference in the probabilities of t1 and t2 will be reduced
which will also reduce the variance

Imagine that we have two different trajectories in our batch. The first has a positive total reward and the second has
a negative one. In this case, it's easy for the PG to update the policy such that the first trajectory's actions get
higher probability. However, if we add a constant value to our reward function, for example such that the second one
receives zero reward, this time the gradient will be completely different since it will not reduce the probabilities for
the sends trajectory at all.

Adding a baseline fixes this dependence on absolute values of the rewards, since we'll compare all rewards to some
constant number. In the example above, subtracting a baseline will cause the gradient to be the same in both cases.
"""


part1_q2 = r"""
$v_\pi(s)$ = $max_aq_\pi(s,a)$ - The value function is the upper bound on the q-values from current state so it is a 
valid approximation of "how far" is the reward from the mean reward value function from the state.
"""


part1_q3 = r"""
1. In the results from the first part we can see that for the reward vpg < bpg < epg < cpg, so the highest result that
we got was for the model that used both baseline and entropy losses which is what we expected - the loss takes into 
account both baseline that reduces variance and entropy which encourages exploration in the moves that the agent takes.
The lowest reward we got was for the vanilla model as we expected since its loss function is not as good as the other 
ones. We can also see that the reward for bpg < epg so it seems that the entropy is important, but the gap is not so
big so it is not significant.
2. In the results from the second part we can see that the aac model got the best results - it used both entropy loss
and a value function approximator network which lead to better performance. The policy model in this part was simpler - 
it had only 2 hidden layers instead of 3 as in other models and it also has fewer parameters and still the results are 
better and that indicates that the "smart" baseline that is approximated by the v_net boosts the total reward.
"""
