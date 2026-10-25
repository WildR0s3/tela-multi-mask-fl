from tela import TELA
import torch
from logger.logging_module import initilaize_logger, log
from curiosity_replay_buffer import CuriosityReplayBuffer
import gymnasium as gym
from maps import Maps
import os

## used for more precide CUDA debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# SEQ_LENGTH = 11
# EPISODES = 32
# EPOCHS = 32

# TODO - L=−E[Log(Probability of Action)×Reward], for reward shaping try RND (Random network distilation)

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
            render_mode="human"  # human or rgb_array
            )


    def run_model(self):

        if self.args.batch_size < 2 or self.args.batch_size % 2 != 0:
            raise ValueError("batch_size must be an even number of at least 2")

        replay_count = self.args.batch_size // 2
        generated_count = self.args.batch_size - replay_count
        replay_buffer = CuriosityReplayBuffer(capacity=replay_count)
        model_tela = TELA(self.args).to(self.args.device)

        for episode_index in range(self.args.episodes):
            state, info = self.env.reset()
            collected_batches = []
            collected_actions = []

            warmup_count = replay_count if len(replay_buffer) == 0 else 0
            rollout_count = generated_count + warmup_count
            for _ in range(rollout_count):
                actions = torch.randint(self.args.action_dim, (4,), dtype=torch.long)
                real_states_batch = torch.full((7,), self.args.pad_token, dtype=torch.long)
                real_states_batch[0] = self.args.start_token
                real_states_batch[1] = state

                for step, action in enumerate(actions):
                    new_state, reward, terminated, truncated, info = self.env.step(action.item()) 
                    real_states_batch[step+2] = new_state

                    if terminated or truncated:
                        if step < 3:
                            real_states_batch[step+3] = self.args.term_token
                            real_states_batch[step+4:] = self.args.pad_token
                        else:
                            real_states_batch[6] = self.args.term_token
                        break
                    if step == 3:
                        real_states_batch[6] = self.args.term_token

                state = new_state
                collected_batches.append(real_states_batch)
                collected_actions.append(actions)
                if terminated or truncated:
                    state, info = self.env.reset()

            log(f"==========\tEpisode {episode_index+1} END\t==========")
            generated_states = torch.stack(collected_batches, dim=0).to(self.args.device)
            generated_actions = torch.stack(collected_actions, dim=0).to(self.args.device)

            if warmup_count:
                replay_buffer.update(model_tela, generated_actions[:replay_count],
                                     generated_states[:replay_count],)
                generated_actions = generated_actions[replay_count:]
                generated_states = generated_states[replay_count:]

            replay_actions, replay_states = replay_buffer.sample(replay_count, self.args.device)
            epoch_real_rand_actions_batches = torch.cat((generated_actions, replay_actions), dim=0)
            epoch_real_states_batches = torch.cat((generated_states, replay_states), dim=0)
            valid_token_mask = model_tela.build_valid_token_mask(epoch_real_states_batches, self.args)

            if episode_index % 100 == 0:
                log("Generated actions:")
                log(f"\n{generated_actions}")
                log("Training states (generated then replay):")
                log(f"\n{epoch_real_states_batches}")

            for epoch_index in range(self.args.epochs):
                logits_states, logits_actions, mask, p_mask, targets = model_tela.forward(
                    epoch_real_rand_actions_batches, epoch_real_states_batches,valid_token_mask,)
                model_tela.learn_model(logits_states, logits_actions,
                                       targets, mask, valid_token_mask, epoch_index,)

            replay_buffer.update(model_tela, epoch_real_rand_actions_batches, epoch_real_states_batches,)
            if replay_buffer.entries:
                log(
                    f"Replay buffer: {len(replay_buffer)} sequences, "
                    f"hardest score: {replay_buffer.entries[0].score:.4f}"
                )

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
            log(f"Test loss: {results_dict['loss']}, state_loss: {results_dict['state_loss']}, action_loss: {results_dict['action_loss']}")

            start_state_example = torch.tensor([0])
            end_state_example = torch.tensor([5])

            mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
            actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm, args=self.args)
            log(f"actions: {actions_pred}")
            log(f"states: {states_pred}")

            # TODO - call those 2 functions

        # torch.save(model_tela.state_dict(), r'res\tela_weights.pth')

        # model_tela = TELA().to(self.args.device)
        # model_tela.load_state_dict(torch.load(r'res\tela_weights.pth', map_location=self.args.device))

        start_state_example = torch.tensor([0])
        end_state_example = torch.tensor([5])

        mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
        actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm, args=self.args)
        log(f"actions: {actions_pred}")
        log(f"states: {states_pred}")

        start_state_example = torch.tensor([0])
        end_state_example = torch.tensor([12])

        mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
        actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm, args=self.args)
        log(f"actions: {actions_pred}")
        log(f"states: {states_pred}")

        start_state_example = torch.tensor([5])
        end_state_example = torch.tensor([2])

        mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
        actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm, args=self.args)
        log(f"actions: {actions_pred}")
        log(f"states: {states_pred}")
            # batches : list[torch.tensor] = model_tela.batch_preparation(epoch_real_rand_actions_batches, epoch_real_states_batches)


            # for epoch in range(EPOCHS):
            #     for batch_idx 

    
    




