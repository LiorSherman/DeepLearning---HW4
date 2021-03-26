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
Subtracting a baseline helps reducing the variance of the policy gradient because it reduces the weight-term in a way
that makes the expression lean towards the zero by making it roughly about the right scale, which results in a reduced
variance by definition. 
For example when there is a reward of 1,000,000 and the next moment there is a reward of 999,000, and we have to "kind 
of" adjust in direction where we're always moving of policies parameters up but it's just the relative difference 
that determines how much up you should make them up. It would be much convinient to have them rescaled so we will have 
something like plus/minus one instead.
What is amazing, is that this trick as stated doesn't effect the expected value of the pg, and results in a much
faster convergence.
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
