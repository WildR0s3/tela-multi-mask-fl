from tela import TELA
import torch
from logger.logging_module import initilaize_logger, log
from curiosity_replay_buffer import CuriosityReplayBuffer
from planner_replay_buffer import PlannerReplayBuffer
from reward_planner import RewardPlanner
import gymnasium as gym
from maps import Maps
import math
import os

## used for more precide CUDA debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# SEQ_LENGTH = 11
# EPISODES = 32
# EPOCHS = 32

# depreciated TODO - L=−E[Log(Probability of Action)×Reward], for reward shaping try RND (Random network distilation)
# TODO fill planner only with sequences leading to reward (in the future since there might be many sources of reward planner might be running couple of rounds, to be precised)
# TODO checked as well full 3-step generation what is current loss of it and monitor that as "suprise"
# TODO planner should collect sequences of rewards only, and he should be learning only "rewarding sequences", but preferably more than one to have multiple paths to rewards. 
#      What in the case of multiple rewards on the board? Should getting to a reward every time not use that reward since prediction error is zero?


class Training():

    def __init__(self, args):
        self.args = args
        initilaize_logger()
        os.environ['SDL_VIDEO_WINDOW_POS'] = '-1280,360' # position gymnasium render window for multi monitor setup

        maps = {
            "4x4" : Maps.clean_4x4,
            "7x7" : Maps.clean_7x7
        }
    
        self.env = gym.make(
            "FrozenLake-v1",
            desc=maps[args.map],
            max_episode_steps=64, 
            is_slippery=False, 
            render_mode=self.args.render_mode  # human or rgb_array
        )



    def run_model(self):

        if self.args.batch_size < 2 or self.args.batch_size % 2 != 0:
            raise ValueError("batch_size must be an even number of at least 2")

        replay_count = self.args.batch_size // 2
        generated_count = self.args.batch_size - replay_count
        replay_buffer = CuriosityReplayBuffer(capacity=replay_count)
        planner_batch_size = 32 # max(1, self.args.batch_size // 4)
        planner_generated_count = math.ceil(generated_count / 4)
        planner_buffer = PlannerReplayBuffer()
        model_tela = TELA(self.args).to(self.args.device)
        model_planner = RewardPlanner(self.args).to(self.args.device)

        for episode_index in range(self.args.episodes):
  
            state, info = self.env.reset()
            
            collected_batches = []
            collected_actions = []
            collected_rewards = []
            collected_episode_ends = []

            warmup_count = replay_count if len(replay_buffer) == 0 else 0
            rollout_count = generated_count + warmup_count

            for _ in range(rollout_count):
                planner_tokens = model_planner.generate(state, self.args)
                log(f"Planner tokens: {planner_tokens}")
                for end_state in [planner_tokens[0][3].item(), planner_tokens[0][5].item(), planner_tokens[0][7].item(), planner_tokens[0][9].item()]: # [16, 32, 46, 48] planner_tokens[0][3].item(), planner_tokens[0][5].item(), planner_tokens[0][7].item(), planner_tokens[0][9].item()
                    mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
                    actions_projection, states_pred = model_tela.generate_emigm_3steps(torch.tensor([state]), torch.tensor([end_state]), mask_emigm, args=self.args)
                    log(f"actions: {actions_projection}")
                    log(f"states: {states_pred}")

                    # actions = torch.randint(self.args.action_dim, (4,), dtype=torch.long)
                    real_states_batch = torch.full((7,), self.args.pad_token, dtype=torch.long)
                    real_states_batch[0] = self.args.start_token
                    real_states_batch[1] = state
                    achieved_reward = False

                    for step, action in enumerate(actions_projection.flatten()):
                        new_state, reward, terminated, truncated, info = self.env.step(action.item()) 
                        achieved_reward = achieved_reward or reward > 0
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
                    collected_actions.append(actions_projection)
                    collected_rewards.append(achieved_reward)
                    collected_episode_ends.append(terminated or truncated)
                    if terminated or truncated:
                        state, info = self.env.reset()

            log(f"==========\tEpisode {episode_index+1} END\t==========")
            generated_states = torch.stack(collected_batches, dim=0).to(self.args.device)
            generated_actions = torch.stack(collected_actions, dim=0).to(self.args.device)
            generated_rewards = torch.tensor(collected_rewards, dtype=torch.bool, device=self.args.device)
            generated_episode_ends = torch.tensor(collected_episode_ends, dtype=torch.bool, device=self.args.device)

            planner_source_states = generated_states[warmup_count:]
            planner_source_rewards = generated_rewards[warmup_count:]
            planner_source_ends = generated_episode_ends[warmup_count:]
            compressed_states, compressed_rewards = self.collect_planner_sequences(
                planner_source_states.cpu(),
                planner_source_rewards.cpu(),
                planner_source_ends.cpu(),
            )
            planner_candidate_tokens = RewardPlanner.interleave_sequences(
                compressed_states, compressed_rewards, self.args.pad_token, self.args.term_token
            ).to(self.args.device)
            if planner_candidate_tokens.shape[0] > planner_generated_count:
                planner_candidate_tokens = planner_candidate_tokens[:planner_generated_count]

            if warmup_count:
                replay_buffer.update(model_tela, generated_actions[:replay_count],
                                     generated_states[:replay_count],)
                generated_actions = generated_actions[replay_count:]
                generated_states = generated_states[replay_count:]

            replay_actions, replay_states = replay_buffer.sample(replay_count, self.args.device)
            epoch_real_rand_actions_batches = torch.cat((generated_actions, replay_actions), dim=0)
            epoch_real_states_batches = torch.cat((generated_states, replay_states), dim=0)
            valid_token_mask = model_tela.build_valid_token_mask(epoch_real_states_batches, self.args)

            planner_tokens = planner_buffer.prepare_batch(
                model_planner,
                planner_candidate_tokens,
                planner_batch_size,
                self.args.device,
            )

            if episode_index % 100 == 0:
                log("Generated actions:")
                log(f"\n{generated_actions}")
                log("Training states (generated then replay):")
                log(f"\n{epoch_real_states_batches}")

            planner_loss = None
            for epoch_index in range(self.args.epochs):
                logits_states, logits_actions, mask, p_mask, targets = model_tela.forward(
                    epoch_real_rand_actions_batches, epoch_real_states_batches, valid_token_mask,)
                model_tela.learn_model(logits_states, logits_actions,
                                       targets, mask, valid_token_mask, epoch_index,)
                if planner_tokens.shape[0]:
                    planner_loss = model_planner.learn_model(planner_tokens)

            planner_buffer.update(model_planner, planner_candidate_tokens)
            best_planner_score = planner_buffer.best_score
            planner_loss_text = f"{planner_loss:.4f}" if planner_loss is not None else "n/a"
            best_planner_score_text = (
                f"{best_planner_score:.4f}" if best_planner_score is not None else "n/a"
            )
            log(
                f"Planner batch: {planner_tokens.shape[0]} sequences, "
                f"loss={planner_loss_text}, "
                f"replay: {planner_buffer.successful_count} successful/"
                f"{planner_buffer.unique_count} unique, best score={best_planner_score_text}"
            )

            replay_buffer.update(model_tela, epoch_real_rand_actions_batches, epoch_real_states_batches,)
            if replay_buffer.entries:
                log(
                    f"Replay buffer: {len(replay_buffer)} sequences, "
                    f"hardest score: {replay_buffer.entries[0].score:.4f}"
                )

            # action_tensor_1 = torch.tensor([[0, 1, 1, 1],
            #                                 [3, 0, 2, 3],
            #                                 [2, 3, 0, 2],
            #                                 [1, 0, 2, 2]]).to(self.args.device)
            # state_tensor_1 = torch.tensor([[16,  0,  0,  4,  8, 12, 17],
            #                                 [16, 12,  8,  8,  9,  5, 17],
            #                                 [16,  5,  6,  2,  1,  2, 17],
            #                                 [16,  2,  6,  5,  6,  7, 17]]).to(self.args.device)
            # fixed_mask = torch.tensor([[False, False,  True,  True,  True,  True, False, False,  True,  True, False],
            #                             [False, False,  True,  True,  True,  True,  True, False,  True, False, False],
            #                             [False, False, False,  True,  True, False,  True,  True,  True, False, False],
            #                             [False, False, False,  True, False,  True,  True, False, False,  True, False]]).to(self.args.device)
            # results_dict = model_tela.test_loss(actions=action_tensor_1, states=state_tensor_1, fixed_mask=fixed_mask)
            # log(f"Test loss: {results_dict['loss']}, state_loss: {results_dict['state_loss']}, action_loss: {results_dict['action_loss']}")

            # start_state_example = torch.tensor([0])
            # end_state_example = torch.tensor([5])

            # mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
            # actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm, args=self.args)
            # log(f"actions: {actions_pred}")
            # log(f"states: {states_pred}")

            # TODO - call those 2 functions

        torch.save(model_tela.state_dict(), rf'res\tela_weights_{self.args.map}.pth')
        torch.save(model_planner.state_dict(), rf"res\reward_planner_{self.args.map}.pth")

        # model_tela = TELA().to(self.args.device)
        # model_tela.load_state_dict(torch.load(r'res\tela_weights.pth', map_location=self.args.device))

        start_state_example = torch.tensor([0])
        end_state_example = torch.tensor([5])

        mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)
        actions_pred, states_pred = model_tela.generate_emigm_3steps(start_state_example, end_state_example, mask_emigm, args=self.args)
        log(f"actions: {actions_pred}")
        log(f"states: {states_pred}")

        planner_tokens = model_planner.generate(start_state_example, self.args)
        log(f"Planner tokens: {planner_tokens}")

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

        # state, info = self.env.reset()
        # for end_state in [5, 7, 13, 15]:
        #     actions_pred, states_pred = model_tela.generate_emigm_3steps(torch.tensor([state]), torch.tensor([end_state]), mask_emigm, args=self.args)
        #     for action in actions_pred.flatten():
        #         new_state, reward, terminated, truncated, info = self.env.step(action.item())
        #     state = new_state 


    def collect_planner_sequences(self, actor_states, achieved_rewards, episode_ends):
        planner_states = []
        planner_rewards = []
        actor_count = actor_states.shape[0]

        for group_start in range(0, actor_count, 4):
            states = torch.full((7,), self.args.pad_token, dtype=torch.long)
            rewards = torch.full_like(states, self.args.pad_token)
            states[0] = self.args.start_token
            rewards[0] = self.args.start_token
            group_end = min(group_start + 4, actor_count)
            states[1] = actor_states[group_start, 1]
            rewards[1] = RewardPlanner.NO_REWARD_TOKEN
            position = 2
            expected_start = int(states[1].item())
            ended = False

            for row_index in range(group_start, group_end):
                actor_sequence = actor_states[row_index]
                if int(actor_sequence[1].item()) != expected_start:
                    break

                if episode_ends[row_index]:
                    terminal_positions = torch.nonzero(
                        actor_sequence[2:6] == self.args.term_token, as_tuple=False
                    ).flatten()
                    endpoint_index = int(terminal_positions[0].item()) + 1 if terminal_positions.numel() else 5
                else:
                    endpoint_index = 5

                states[position] = actor_sequence[endpoint_index]
                rewards[position] = (
                    RewardPlanner.REWARD_TOKEN
                    if achieved_rewards[row_index]
                    else RewardPlanner.NO_REWARD_TOKEN
                )
                expected_start = int(actor_sequence[endpoint_index].item())
                position += 1

                if episode_ends[row_index]:
                    ended = True
                    break

            if position < 7:
                states[position] = self.args.term_token
                rewards[position] = self.args.term_token
                position += 1
            states[position:] = self.args.pad_token
            rewards[position:] = self.args.pad_token
            planner_states.append(states)
            planner_rewards.append(rewards)

        return torch.stack(planner_states), torch.stack(planner_rewards)



    def test_model(self):

        model_tela = TELA(self.args).to(self.args.device)
        model_tela.load_state_dict(torch.load(rf'res\tela_weights_{self.args.map}.pth', map_location=self.args.device))
        mask_emigm = torch.ones(self.args.seq_length, dtype=torch.float, device=self.args.device)

        model_planner = RewardPlanner(self.args).to(self.args.device)
        model_planner.load_state_dict(torch.load(rf'res\reward_planner_{self.args.map}.pth', map_location=self.args.device))

        state, info = self.env.reset()
        planner_tokens = model_planner.generate(torch.tensor([state]), self.args)
        log(f"Planner tokens: {planner_tokens}")
        for end_state in [planner_tokens[0][3].item(), planner_tokens[0][5].item(), planner_tokens[0][7].item(), planner_tokens[0][9].item()]: # [16, 32, 46, 48] planner_tokens[0][3].item(), planner_tokens[0][5].item(), planner_tokens[0][7].item(), planner_tokens[0][9].item()
            actions_pred, states_pred = model_tela.generate_emigm_3steps(torch.tensor([state]), torch.tensor([end_state]), mask_emigm, args=self.args)
            log(f"actions: {actions_pred}")
            log(f"states: {states_pred}")
            for action in actions_pred.flatten():
                new_state, reward, terminated, truncated, info = self.env.step(action.item())
            state = new_state 
    
    


