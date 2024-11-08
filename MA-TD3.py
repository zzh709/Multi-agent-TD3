import os
import torch
import random
import numpy as np
from matplotlib import animation
from environment import Environment
from matplotlib import pyplot as plt
from network import AttentionActor, AttentionCritic, Critic
from collections import namedtuple, deque
# import torch.nn.DataParallel as DP


device = "cuda" if torch.cuda.is_available() else "cpu"
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.as_tensor(state).to(torch.float32).to(device)
        action = torch.as_tensor(action).to(torch.float32).to(device)
        reward = torch.as_tensor(reward).to(torch.float32).to(device)
        next_state = torch.as_tensor(next_state).to(torch.float32).to(device)
        done = torch.as_tensor(done).to(torch.float32).to(device)

        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, n_agents, state_dim, observe_dim, action_dim, hidden_dim):
        self.actor = AttentionActor(n_agents, observe_dim, action_dim, hidden_dim).to(device)
        self.target_actor = AttentionActor(n_agents, observe_dim, action_dim, hidden_dim).to(device)

        self.critic1 = Critic(n_agents, state_dim, action_dim, 128, 64).to(device)
        self.target_critic1 = Critic(n_agents, state_dim, action_dim, 128, 64).to(device)

        self.critic2 = Critic(n_agents, state_dim, action_dim, 128, 64).to(device)
        self.target_critic2 = Critic(n_agents, state_dim, action_dim, 128, 64).to(device)

        self.update_target_actor(1.0)
        self.update_target_critic(1.0)

    def update_target_actor(self, tau):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update_target_critic(self, tau):
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    @torch.no_grad()
    def exploit(self, observe):
        actions = self.actor(observe)
        return actions

    @torch.no_grad()
    def explore(self, observe):
        actions = self.actor(observe)
        actions = actions + torch.randn(actions.shape, device=device) * 0.1
        return torch.clip(actions, -0.5, 0.5)


class Trainer:
    def __init__(self, agents: list[Agent], env: Environment, memory_size=1e6, gamma=0.95):
        self.env = env
        self.agents = agents

        self.gamma = gamma
        self.memory = ReplayMemory(int(memory_size))

        self.actor_loss = 0
        self.critic_loss = 0

        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []

        self.actor_optimizer = torch.optim.Adam(self.agents[0].actor.parameters(), 3e-4)
        self.critic_optimizer = torch.optim.Adam(list(self.agents[0].critic1.parameters()) + list(self.agents[0].critic2.parameters()), 3e-4)

    def get_actor_parameters(self):
        param = []
        for agent in self.agents:
            param += list(agent.actor.parameters())
        return param

    def get_critic_parameters(self):
        param = []
        for agent in self.agents:
            param += list(agent.critic1.parameters())
            param += list(agent.critic2.parameters())
        return param

    def update_target_actor(self, tau):
        for agent in self.agents:
            agent.update_target_actor(tau)

    def update_target_critic(self, tau):
        for agent in self.agents:
            agent.update_target_critic(tau)

    def get_observe(self, agent_id, state):
        observe = state[..., self.env.state_dim - self.env.observe_dim:]

        next_agent_id = (agent_id + 1) % len(self.agents)
        other_agent_id = [i for i in range(len(self.agents)) if i not in [agent_id, next_agent_id]]
        return torch.reshape(observe[..., [agent_id, next_agent_id] + other_agent_id, :], (-1, state.shape[-2], state.shape[-1]))

    @torch.no_grad()
    def exploit(self, state):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.exploit(self.get_observe(i, state))
            actions.append(action)
        actions = torch.cat(actions, 0)
        return actions.detach().to("cpu").numpy()

    @torch.no_grad()
    def explore(self, state):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.explore(self.get_observe(i, state))
            actions.append(action)
        actions = torch.cat(actions, 0)
        return actions.detach().to("cpu").numpy()

    def learn(self, step, transitions):
        batch = Transition(*zip(*transitions))

        states_batch = torch.stack(batch.state, 0)
        actions_batch = torch.stack(batch.action, 0)
        rewards_batch = torch.stack(batch.reward, 0)
        next_states_batch = torch.stack(batch.next_state, 0)
        done_batch = torch.stack(batch.done, 0)

        # rewards_batch /= torch.std(rewards_batch, 0, keepdim=True)

        # 计算Q(target)
        with torch.no_grad():
            next_actions_batch = []
            for i, agent in enumerate(self.agents):
                next_observe_batch = self.get_observe(i, next_states_batch)
                next_actions_batch.append(agent.target_actor(next_observe_batch))
            next_actions_batch = torch.stack(next_actions_batch, 1)
            next_actions_batch = torch.clip(next_actions_batch + torch.clip(torch.randn(next_actions_batch.shape, device=device) * 0.1, -0.01, 0.01), -0.5, 0.5)

            target_q_value_batch = []
            for i, agent in enumerate(self.agents):
                next_q_value_batch1 = agent.target_critic1(next_states_batch, next_actions_batch).detach()
                next_q_value_batch2 = agent.target_critic2(next_states_batch, next_actions_batch).detach()
                next_q_value_batch = torch.minimum(next_q_value_batch1, next_q_value_batch2)
                target_q_value_batch.append(next_q_value_batch * (1 - done_batch[:, i]) * self.gamma + rewards_batch[:, i])
            target_q_value_batch = torch.stack(target_q_value_batch, 1)

        q_value_batch1 = torch.stack([agent.critic1(states_batch, actions_batch) for agent in self.agents], 1)
        q_value_batch2 = torch.stack([agent.critic2(states_batch, actions_batch) for agent in self.agents], 1)
        critic_loss = 0.5 * (torch.nn.functional.mse_loss(q_value_batch1, target_q_value_batch) + torch.nn.functional.mse_loss(q_value_batch2, target_q_value_batch))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agents[0].critic1.parameters()) + list(self.agents[0].critic2.parameters()), 10.0)
        self.critic_optimizer.step()
        # self.total_critic_scheduler.step(critic_loss)

        self.update_target_critic(0.01)
        self.critic_loss += (critic_loss.item() - self.critic_loss) / step

        # 更新 Actor。
        if step % 2 == 0:
            new_action_batch = torch.stack([agent.actor(self.get_observe(i, states_batch)) for i, agent in enumerate(self.agents)], 1)
            actor_loss = -torch.mean(torch.stack([agent.critic1(states_batch, new_action_batch) for i, agent in enumerate(self.agents)], 1)) + torch.mean(torch.square(new_action_batch)) * 1e-3

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[0].actor.parameters(), 10.0)
            self.actor_optimizer.step()

            self.update_target_actor(0.01)
            self.actor_loss += (actor_loss.item() - self.actor_loss) / (step / 2)

    def train(self, batch_size=512, max_episode=10000):
        step = 0
        all_paths = []  
        for episode in range(max_episode):
            done = 0
            state = self.env.reset()
            path1 = []  
            total_reward = 0
            while np.all(done == 0):
                action = self.explore(torch.tensor(state, dtype=torch.float32, device=device))
                reward, next_state, done = self.env.evolution(action, False)

                reward = np.sum(reward, -1)
                self.memory.push(state, action, reward, next_state, done)

                state = next_state
                # 记录路径信息
                path1.append((state[0,0], state[0,1],state[0,2]))



                total_reward += np.mean(reward)
                if len(self.memory) < batch_size:
                    continue

                step += 1
                self.learn(step, self.memory.sample(batch_size))

                if step % 1000 == 0:
                    self.save("MATD3_2/weights")
                    reward, length = self.eval(True)

                    print("step:%d, reward = %.3f, length = %d, actor_loss = %.3f, critic_loss = %.3f" %
                          (step, reward, length, self.actor_loss, self.critic_loss))
                    print(state[:, 5])
                    with open("errors.csv", "w") as f:
                        f.write(f"Episode {state[:, 5]} path:\n")


            all_paths.append(path1)
            self.rewards.append(total_reward)
            self.actor_losses.append(self.actor_loss)
            self.critic_losses.append(self.critic_loss)

            if (episode + 1) % 1000 == 0:
                self.plot_loss_reward()
                #self.output_paths(all_paths)

    def output_paths(self, all_paths):

        for episode_idx, path1 in enumerate(all_paths):
            print(f"Episode {episode_idx + 1} path:")
            for step_idx, (acceleration,velocities,positions) in enumerate(path1):
                print(f"Step {step_idx + 1}: acceleration = {acceleration}, Velocities = {velocities}, Positions = {positions}")

        with open("paths.csv", "w") as f:
            for episode_idx, path1 in enumerate(all_paths):
                f.write(f"Episode {episode_idx + 1} path:\n")
                for step_idx, (acceleration,velocities,positions) in enumerate(path1):
                    f.write(f"Step {step_idx + 1}: acceleration = {acceleration}, Velocities = {velocities}, Positions = {positions}")


    @torch.no_grad()
    def eval(self, plot_gif=False):
        done = 0
        total_reward = 0
        state = self.env.reset()

        trajectory = [[state, self.env.get_trajectory()]]
        for agent in self.agents:
            agent.hidden = agent.next_hidden = None
        while np.all(done == 0):
            action = self.exploit(torch.tensor(state, dtype=torch.float32, device=device))
            reward, next_state, done = self.env.evolution(action)
            state = next_state

            total_reward += reward
            trajectory.append([state, self.env.get_trajectory()])

        if plot_gif:
            fig, ax = plt.subplots()

            def plot_animate(index):
                ax.clear()
                for i in range(len(self.agents)):
                    ax.scatter(trajectory[index][0][i, -6], trajectory[index][0][i, -5])
                    ax.plot(trajectory[index][1][i, :, 0], trajectory[index][1][i, :, 1])
                ax.scatter(trajectory[index][0][0, -4], trajectory[index][0][0, -3], marker='*')
                ax.set_xlim(0, 1.5)
                ax.set_ylim(0, 1.2)
                ax.set_title('Step = ' + str(index))

            ani = animation.FuncAnimation(fig, plot_animate, frames=len(trajectory), interval=500)
            ani.save('MATD3_2/result.gif', writer='pillow')

            plt.close(fig)

        return total_reward, len(trajectory) - 1

    def plot_loss_reward(self):
        actor_loss = np.array(self.actor_losses)
        critic_loss = np.array(self.critic_losses)

        plt.clf()

        plt.title("Actor/Critic Loss")
        plt.plot(np.arange(len(self.actor_losses)), actor_loss, label="Actor Loss")
        plt.plot(np.arange(len(self.critic_losses)), critic_loss, label="Critic Loss")
        plt.legend()
        plt.xlabel("Episode")
        plt.savefig("MATD3_2/images/loss.pdf", dpi=600, format="pdf", bbox_inches="tight")

        plt.clf()

        reward = np.array(self.rewards)
        plt.title("Reward")
        plt.plot(np.arange(len(self.rewards)), reward)
        plt.xlabel("Episode")
        plt.savefig("MATD3_2/images/reward.pdf", dpi=600, format="pdf", bbox_inches="tight")

        plt.clf()

        np.save("MATD3_2/reward.npy", reward)
        np.save("MATD3_2/actor_loss.npy", actor_loss)
        np.save("MATD3_2/critic_loss.npy", critic_loss)

    def save(self, path):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), os.path.join(path, f"agent_{i}_actor.pth"))
            torch.save(agent.critic1.state_dict(), os.path.join(path, f"agent_{i}_critic1.pth"))
            torch.save(agent.critic2.state_dict(), os.path.join(path, f"agent_{i}_critic2.pth"))
            torch.save(agent.target_actor.state_dict(), os.path.join(path, f"agent_{i}_target_actor.pth"))
            torch.save(agent.target_critic1.state_dict(), os.path.join(path, f"agent_{i}_target_critic1.pth"))
            torch.save(agent.target_critic2.state_dict(), os.path.join(path, f"agent_{i}_target_critic2.pth"))

    def load(self, path):
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(os.path.join(path, f"agent_{i}_actor.pth")))
            agent.critic1.load_state_dict(torch.load(os.path.join(path, f"agent_{i}_critic1.pth")))
            agent.critic2.load_state_dict(torch.load(os.path.join(path, f"agent_{i}_critic2.pth")))
            agent.target_actor.load_state_dict(torch.load(os.path.join(path, f"agent_{i}_target_actor.pth")))
            agent.target_critic1.load_state_dict(torch.load(os.path.join(path, f"agent_{i}_target_critic1.pth")))
            agent.target_critic2.load_state_dict(torch.load(os.path.join(path, f"agent_{i}_target_critic2.pth")))


def main():
    n_agents = 3
    env_width = 3
    env_height = 3
    env_evo_dt = 0.1

    batch_size = 512
    max_episode = 100000000

    environment = Environment(n_agents, env_width, env_height, env_evo_dt)
    #agents = [Agent(n_agents, environment.state_dim, 2 * environment.observe_dim, environment.action_dim, 128) for _ in range(n_agents)]
    #trainer = Trainer(agents, environment, gamma=0.99)


    agents = Agent(n_agents, environment.state_dim, environment.observe_dim, environment.action_dim, 128)
    trainer = Trainer([agents for _ in range(n_agents)], environment, gamma=0.99)

    #trainer.load('MATD3_2/weights/')
    trainer.train(batch_size, max_episode)


if __name__ == '__main__':
    main()
