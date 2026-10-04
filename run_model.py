from tela import TELA
import torch
from logger.logging_module import initilaize_logger, log
from masks import Masks
from curiosity_buffor import CuriosityBuffor, CBPayload
import gymnasium as gym
from maps import Maps
import math, os

## used for more precide CUDA debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# SEQ_LENGTH = 11
# EPISODES = 32
# EPOCHS = 32

class Training():

    def __init__(self, args):
        self.args = args
        initilaize_logger()
        os.environ['SDL_VIDEO_WINDOW_POS'] = '-1280,360' # position gymnasium render window for multi monitor setup
    
        self.env = gym.make(
            "FrozenLake-v1",
            desc=Maps.clean_4x4,
            max_episode_steps=16, 
            is_slippery=False, 
            render_mode="rgb_array"  # human or rgb_array
            )


    def run_model(self):

        state_dim = self.env.observation_space.n 
        LOSS_BASELINE = math.log(state_dim)
        cb = CuriosityBuffor(state_dim, LOSS_BASELINE)
        model_tela = TELA(self.args).to(self.args.device)

        for i in range(self.args.episodes):
            state, info = self.env.reset()
            done = False

            state_logits_batch = torch.zeros(self.args.batch_size, 4, 18, dtype=torch.float32) # probably it cannot be intialized to zeros, since it is a valid number
            action_logits_batch = torch.zeros(self.args.batch_size, 4, 4)  # probably it cannot be intialized to zeros, since it is a valid number
            collected_batches = []
            collected_actions = []

            counter = 1
            for j in range(self.args.batch_size):
                actions = cb.select_actions(state)
                real_states_batch = torch.zeros(7, dtype=torch.long) 
                real_states_batch[0] =  16 # START
                real_states_batch[1] = state

                for step, action in enumerate(actions):
                    new_state, reward, terminated, truncated, info = self.env.step(action.item()) 
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
                    state, info = self.env.reset()
                    done = False
                if counter == 4:
                    done = True

                counter += 1

            log(f"==========\tEpisode {i+1} END\t==========")
            epoch_real_states_batches = torch.stack(collected_batches, dim=0)
            epoch_real_states_batches[epoch_real_states_batches == -1] = 18
            epoch_real_states_batches = epoch_real_states_batches.to(self.args.device)
            epoch_real_rand_actions_batches = torch.stack(collected_actions, dim=0)
            epoch_real_rand_actions_batches = epoch_real_rand_actions_batches.to(self.args.device)
            valid_token_mask = model_tela.build_valid_token_mask(epoch_real_states_batches)
            log("Real actions")
            log(f"\n{epoch_real_rand_actions_batches}")
            log("Real states:")
            log(f"\n{epoch_real_states_batches}")

            for i in range(self.args.epochs):
                logits_states, logits_actions, mask, p_mask, targets = model_tela.forward(epoch_real_rand_actions_batches, epoch_real_states_batches, valid_token_mask)
                predictions_states = torch.argmax(logits_states, dim=2)
                predictions_actions = torch.argmax(logits_actions, dim=2)
                model_tela.learn_model(logits_states, logits_actions, targets, mask, valid_token_mask, i)

            action_tensor_1 = torch.tensor([[0, 1, 1, 1],
                                            [3, 0, 2, 3],
                                            [2, 3, 0, 2],
                                            [1, 0, 2, 2]]).to(self.args.device)
            state_tensor_1 = torch.tensor([[16,  0,  0,  4,  8, 12, 17],
                                            [16, 12,  8,  8,  9,  5, 17],
                                            [16,  5,  6,  2,  1,  2, 17],
                                            [16,  2,  6,  5,  6,  7, 17]]).to(self.args.device)
            fixed_mask = torch.tensor([[False, False,  True,  True,  True,  True, False, False,  True,  True, False],
                                        [False, False,  True,  True,  True,  True,  True, False,  True, False, False],
                                        [False, False, False,  True,  True, False,  True,  True,  True, False, False],
                                        [False, False, False,  True, False,  True,  True, False, False,  True, False]]).to(self.args.device)
            results_dict = model_tela.test_loss(actions=action_tensor_1, states=state_tensor_1, fixed_mask=fixed_mask)
            log(f"Test loss: {results_dict["loss"]}, state_loss: {results_dict["state_loss"]}, action_loss: {results_dict["action_loss"]}")

            # TODO - call those 2 functions

        torch.save(model_tela.state_dict(), r'res\tela_weights.pth')


        # model_tela = TELA().to(self.args.device)
        # model_tela.load_state_dict(torch.load(r'res\tela_weights.pth', map_location=self.args.device))

        start_state_example = torch.tensor([0])
        end_state_example = torch.tensor([5])


        mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
        actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm)
        log(f"actions: {actions_pred}")
        log(f"states: {states_pred}")
            # batches : list[torch.tensor] = model_tela.batch_preparation(epoch_real_rand_actions_batches, epoch_real_states_batches)


            # for epoch in range(EPOCHS):
            #     for batch_idx 

    
    




