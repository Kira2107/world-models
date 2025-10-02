"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
# We switch to gymnasium for modern API compatibility (5-tuple return)
import gymnasium as gym
import numpy as np
# Assuming utils.misc and sample_continuous_policy are available
from utils.misc import sample_continuous_policy

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    # Check if the output directory exists
    assert exists(data_dir), "The data directory does not exist..."

    # Initialize the environment using the Gymnasium standard (v3) 
    # and specify render_mode for headless data collection.
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    seq_len = 1000

    for i in range(rollouts):
        # Reset the environment for a new rollout. Gymnasium reset() returns (observation, info)
        s, info = env.reset() # Capture the initial observation
        
        # We need to manually add the first observation to s_rollout
        s_rollout = [s] # states (observations)

        # REMOVED: env.env.viewer.window.dispatch_events() (Fix for previous AttributeError)

        if noise_type == 'white':
            # White noise policy (random actions)
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            # Brown noise policy (smoother random walk actions)
            # This returns a list of action arrays/lists
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)
            
        # FIX 1: Convert the list of actions to a single NumPy array with explicit float32 dtype.
        a_rollout = np.array(a_rollout, dtype=np.float32)

        r_rollout = [] # rewards
        d_rollout = [] # done flags (terminals: terminated or truncated)

        t = 0
        while True:
            # Check if we have actions left in the rollout sequence
            if t >= seq_len:
                break
                
            # FIX 2: Explicitly convert the action slice back to a fresh np.float32 array 
            # immediately before calling step(). This prevents subtle type degradation
            # that occurs when passing NumPy array slices to the Box2D C++ backend.
            action = np.array(a_rollout[t], dtype=np.float32)
            t += 1

            # Execute a step in the environment. Gymnasium returns 5 values.
            # s is observation, r is reward.
            # terminated (True if environment reached a terminal state, e.g., crashed)
            # truncated (True if episode ended due to time limit or max steps)
            # info is auxiliary information.
            s, r, terminated, truncated, info = env.step(action)
            
            # Combine terminated and truncated into a single 'done' flag for old-style compatibility
            done = terminated or truncated
            
            # The 'render' call is no longer required in the loop when render_mode is set in gym.make
            # However, some Gym codebases rely on this call for the observation update.
            # Since the observation 's' is returned by env.step(), we can usually remove it,
            # but we keep it commented out for now in case of custom env behavior.
            # env.render() 

            # Store the data
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]

            # Check for episode termination (done=True)
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                
                # Save the rollout data. a_rollout is already the final float32 NumPy array.
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                          observations=np.array(s_rollout),
                          rewards=np.array(r_rollout),
                          actions=a_rollout[:len(s_rollout)], # Slice actions to match the episode length
                          terminals=np.array(d_rollout))
                break
        
        # Close the environment after each rollout to manage resources, especially when running threads.
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    generate_data(args.rollouts, args.dir, args.policy)
