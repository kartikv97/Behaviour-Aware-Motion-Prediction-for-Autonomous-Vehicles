#
# @copyright 2020. Kartik Venkat, Kushagra Agrawal, Aditya Khopkar, Sanuallah Patan Khan
#
# @file Grip.py
#
# @author Kartik Venkat,
#         Kushagra Agrawal,
#         Aditya Khopkar,
#         Sanuallah Patan Khan
#
# @license BSD Clause 3.
#
# @brief This iis the main demo script for the project.
#
#

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import TensorBoard

import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread

from tqdm import tqdm

# try:
#     sys.path.append(glob.glob('../CARLA_0.9.9.4/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
# import carla

from DQNetwork import DQNAgent
from Agent import CarEnv


from GripPredModel.src.model import Model
from GripPredModel.src.dataProcess import getDirectFrameDict
from GripPredModel.src.Grip import my_load_model, run_test


SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.4
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95 ## 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

dev = 'cuda:0'
data_root = '/home/kartik/Documents/CMSC818B/FinalProject/Behaviour-Aware-Motion-Prediction-for-Autonomous-Vehicles/GripPredModel/'
graph_args = {'max_hop': 2, 'num_node': 120}
pretrained_model_path = data_root + 'models/model_epoch_0049.pt'


if __name__ == '__main__':
    try:

        model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
        model.to(dev)
        model = my_load_model(model, pretrained_model_path)

        FPS = 60
        # For stats
        ep_rewards = [-200]

        # For more repetitive results
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)

        # Memory fraction, used mostly when training multiple agents
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
        backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

        # Create models folder
        if not os.path.isdir('models'):
            os.makedirs('models')

        # Create agent and environment
        agent = DQNAgent()
        env = CarEnv()


        # Start training thread and wait for training to be initialized
        trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
        trainer_thread.start()
        while not agent.training_initialized:
            time.sleep(0.01)

        # Initialize predictions - forst prediction takes longer as of initialization that has to be done
        # It's better to do a first prediction then before we start iterating over episode steps
        agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

        # Iterate over episodes
        for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
            #try:

                env.collision_hist = []

                # Update tensorboard step every episode
                agent.tensorboard.step = episode

                # Restarting episode - reset episode reward and step number
                episode_reward = 0
                step = 1
                frame_count = 0
                frames_list = []

                # Reset environment and get initial state
                current_state = env.reset()

                # Reset flag and start iterating until episode ends
                done = False
                episode_start = time.time()

                # Play for given number of seconds only
                while True:
                    # frame_id, object_id, object_type, position_x, position_y, position_z,
                    # object_length, object_width, object_height, heading
                    while frame_count < 6:
                        extent = env.agent_vehicle.bounding_box.extent
                        transform = env.agent_vehicle.get_transform()
                        frame_id = frame_count
                        object_id = env.agent_vehicle.id
                        object_type = 1
                        position_x, position_y, position_z = transform.location.x, transform.location.y, transform.location.z
                        object_length, object_width, object_height = 2 * extent.x , 2 * extent.y, 2 * extent.z
                        heading = transform.rotation.yaw

                        frames_list.append([frame_id, object_id, object_type,
                                            position_x, position_y, position_z,
                                            object_length, object_width,
                                            object_height, heading])

                        frame_count += 1

                    prediction = env.predict(model, frames_list, graph_args)
                    # This part stays mostly the same, the change is to query a model for Q values
                    if np.random.random() > epsilon:
                        # Get action from Q table
                        action = np.argmax(agent.get_qs(current_state))
                    else:
                        # Get random action
                        action = np.random.randint(0, 3)
                        # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                        time.sleep(1/FPS)

                    new_state, reward, done, _ = env.step(action, prediction)
                    # new_state, reward, done, _ = env.step(0)

                    # Transform new continuous state to new discrete state and count reward
                    episode_reward += reward

                    # Every step we update replay memory
                    agent.update_replay_memory((current_state, action, reward, new_state, done))

                    current_state = new_state
                    step += 1
                    frame_count += 1

                    # if step > 5 or done == True:
                    #     break

                    if step > 1 or done == True:
                        break

                # End of episode - destroy agents
                for actor in env.actor_list:
                    actor.destroy()

                # Append episode reward to a list and log stats (every given number of episodes)
                ep_rewards.append(episode_reward)
                if not episode % AGGREGATE_STATS_EVERY or episode == 1:
                    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

                    # Save model, but only when min reward is greater or equal a set value
                    if min_reward >= MIN_REWARD:
                        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

                # Decay epsilon
                if epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(MIN_EPSILON, epsilon)


        # Set termination flag for training thread and wait for it to finish
        agent.terminate = True
        trainer_thread.join()
        agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        if exit()==0:
            try:
                for actor in env.actor_list:
                    actor.destroy()
                print("Destroyed all agentsa")
            except:
                print("Could not destroy agents")
    except (KeyboardInterrupt,SystemExit):
        try:
            for actor in env.actor_list:
                actor.destroy()
            print("Destroyed all agentsa")
        except:
            print("Could not destroy agents")