import gym
import tensorflow as tf
import numpy as np
import random

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float", [None,4])
        linear = tf.matmul(state, params)
        probabilities = tf.nn.softmax(linear)

    return probabilities, state

def run_episode(env, policy_grad, sess):
    pl_prob, pl_state = policy_grad
    observation = env.reset()
    total_reward = 0

    for _ in xrange(200):
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_prob,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        observation, reward, done, info = env.step(action)

        if done:
            break

    return total_reward




if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    policy_grad = policy_gradient()
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(2000):
        reward = run_episode(env, policy_grad, sess)
        if reward >= 200:
            print i
            break

    t = 0
    for _ in xrange(1000):
        reward = run_episode(env, policy_grad, sess)
        t += reward
    print t / 1000.

