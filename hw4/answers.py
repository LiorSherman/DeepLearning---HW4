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
Using the estimated q-values as regression targets for our state-values gets us a valid approximation because of how 
$v_\pi(s)$ is expressed in terms of $max_aq_\pi(s,a)$ like the hint says.
As we can recall $v_\pi(s)$ = $max_aq_\pi(s,a)$ so we can see that the value function is the maximum of the q-values of
a specific state therefore it is a good approximation of the distance between the reward and the mean reward value
function of the state.
"""


part1_q3 = r"""
1.  The first experiment's results confirm our thoughts of the loss functions.
The best one is the CPG model which utilizes the baseline subtraction that reduces variance and entropy which gave
the agent more freedom to choose moves.
The worst one is the VPG which utilizes none of the above.
The BPG and EPG models are better than the vanilla, each one of them utilizes
one of the improvements and their results are close.
However, we can see from the rewards graph that BPG < EPG but they are close
so we conclude that the difference between them is not that important.

2.  The second experiment's results also confirm our thought of the AAC.
The AAC performs better than the cpg as it used the entropy loss and the advantage function which gave that model the
ability to perform better.
"""
