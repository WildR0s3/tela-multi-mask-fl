from tela import TELA
import torch
from logger.logging_module import initilaize_logger, log
from masks import Masks
from curiosity_buffor import CuriosityBuffor, CBPayload
import gymnasium as gym
from maps import Maps
import math, os
from utils import create_attention_mask
# TODO - add targets in batch preparation most likely
# TODO - in forward make predictions for masked tokens

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEQ_LENGTH = 11
EPISODES = 1
EPOCHS = 10
initilaize_logger()
os.environ['SDL_VIDEO_WINDOW_POS'] = '-1280,360' # position gymnasium render window for multi monitor setup

env = gym.make(
    "FrozenLake-v1",
    desc=Maps.clean_4x4,
    max_episode_steps=16, 
    is_slippery=False, 
    render_mode="human"  # human or rgb_array
    )

state_dim = env.observation_space.n 
LOSS_BASELINE = math.log(state_dim)
cb = CuriosityBuffor(state_dim, LOSS_BASELINE)
model_tela = TELA().to(DEVICE)

for i in range(EPISODES):
    state, info = env.reset()
    done = False

    state_logits_batch = torch.zeros(4, 4, 18, dtype=torch.float32) # probably it cannot be intialized to zeros, since it is a valid number
    action_logits_batch = torch.zeros(4, 4, 4)  # probably it cannot be intialized to zeros, since it is a valid number
    collected_batches = []
    collected_actions = []

    counter = 1
    for j in range(4):
        actions = cb.select_actions(state)
        real_states_batch = torch.zeros(7, dtype=torch.long) 
        real_states_batch[0] =  16 # START
        real_states_batch[1] = state

        for step, action in enumerate(actions):
            new_state, reward, terminated, truncated, info = env.step(action.item()) 
            done = terminated or truncated
            real_states_batch[step+2] = new_state
            
            if done and step == 3:
                real_states_batch[6] = 17 # TERM
            if not done and step == 3: # verify if it is correct to have in all the cases except your testing
                real_states_batch[6] = 17 # TERM
            if step < 3 and terminated:
                real_states_batch[step+3] = 17 # TERM
                real_states_batch[step+4:] = -1
                break

        state = new_state

        collected_batches.append(real_states_batch)
        collected_actions.append(actions)
        if len(collected_batches) == 4:
            done = True
            break
        if terminated:
            state, info = env.reset()
            done = False
        if counter == 4:
            done = True

        counter += 1

    log(f"==========\tEpisode {i+1} END\t==========")
    epoch_real_states_batches = torch.stack(collected_batches, dim=0)
    batch_mask, mask_count = create_attention_mask(epoch_real_states_batches)
    epoch_real_states_batches[epoch_real_states_batches == -1] = 18
    epoch_real_rand_actions_batches = torch.stack(collected_actions, dim=0)
    logits, mask, p_mask, targets = model_tela.forward(epoch_real_rand_actions_batches, epoch_real_states_batches, batch_mask)
    model_tela.learn_model(logits)
    # batches : list[torch.tensor] = model_tela.batch_preparation(epoch_real_rand_actions_batches, epoch_real_states_batches)


    # for epoch in range(EPOCHS):
    #     for batch_idx 

    
    




