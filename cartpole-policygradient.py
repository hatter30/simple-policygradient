import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float", [None,4])
        actions = tf.placeholder("float", [None, 2])
        linear = tf.matmul(state, params)
        probabilities = tf.nn.softmax(linear)
        action_probabilities = tf.reduce_sum(tf.mul(probabilities, actions), reduction_indices=[1])
        advantages = tf.placeholder("float",[None,1])
        eligibility = tf.log(action_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float", [None, 4])
        newvals = tf.placeholder("float", [None, 1])
        w1 = tf.get_variable("w1", [4, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(state,w1)+b1)
        w2 = tf.get_variable("w2", [10,1])
        b2 = tf.get_variable("b2", [1])
        calculated = tf.matmul(h1, w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_prob, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    states = []
    actions = []
    transitions = []
    total_reward = 0
    update_vals = []
    advantages = []

    for _ in xrange(200):
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_prob,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        #record the transition
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        total_reward += reward

        if done:
            break

    # calculate the future reward
    for index, trans in enumerate(transitions):
        observation, action, reward = trans

        # calculate discounted return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[ index + index2 ][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(observation, axis=0)
        current_val = sess.run(vl_calculated, feed_dict={vl_state: obs_vector})[0][0]

        # store the future rewards
        update_vals.append(future_reward)

        # advantages : how much better was this action than current action
        advantages.append(future_reward - current_val)

    # udpate value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages,axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return total_reward

def verify_value_grad(env, value_grad, sess):
    pass




if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    policy_grad = policy_gradient()
    value_grad = value_gradient()
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    t = 0
    reward_list = []
    epoch = 3000
    for i in xrange(epoch):
        reward = run_episode(env, policy_grad, value_grad, sess)
        reward_list.append(reward)
        print "running %4d reward %d" % (i, reward)
        t += reward

    print "average : %d" % (t / epoch)
    plt.plot(reward_list)
    plt.show()


