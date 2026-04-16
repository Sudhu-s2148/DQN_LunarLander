import agent as A
import buffer
import torch
import numpy as np
import gymnasium
import torch.nn as nn
import os

training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gymnasium.make("LunarLander-v2")

experiences = buffer.Buffer()
online_network = A.agent().to(device)
offline_network = A.agent().to(device)

gamma = 0.99
epsilon = 1
decay = 0.997
epsilon_min = 0.05

max_ep = 5000
max_steps = 750
total_steps = 0
total_overall_steps = 0
best_avg_reward = 0

optimizer = torch.optim.Adam(online_network.parameters(), lr=0.0005)

save_path = "best_network.pth"
#X_pos, Y_pos, X_vel, Y_vel,theta, omega, left_contact, right_contact

def bellmans_update(active_network, dormant_network, buffer_exp, gamma, device):
    states_batch = torch.tensor(np.array([t[0] for t in buffer_exp]), dtype=torch.float32).to(device)
    actions_batch = torch.tensor([t[1] for t in buffer_exp], dtype=torch.int64).to(device)
    next_states_batch = torch.tensor(np.array([t[2] for t in buffer_exp]), dtype=torch.float32).to(device)
    rewards_batch = torch.tensor([t[3] for t in buffer_exp], dtype=torch.float32).to(device)
    dones_batch = torch.tensor([t[4] for t in buffer_exp], dtype=torch.float32).to(device)
    with torch.no_grad():
        max_Q = dormant_network.forward(next_states_batch).max(dim = 1).values
        max_Q = max_Q*(1-dones_batch)
        target =rewards_batch + gamma*max_Q
    predicted = active_network.forward(states_batch).gather(1,actions_batch.unsqueeze(1)).squeeze(1)

    loss = nn.MSELoss()(predicted,target)
    return loss

if training:
    for i in range(max_ep):
        step_count = 0
        state,_ = env.reset()
        for j in range(max_steps):
            total_overall_steps+=1
            step_count+=1
            action = online_network.choice(state,epsilon)
            next_state, reward, terminated, truncated,_= env.step(action)
            experiences.push(state,action,next_state,reward,terminated)
            state = next_state
            if len(experiences)>1000:
                exp_batch = experiences.sample(64)
                optimizer.zero_grad()
                loss = bellmans_update(online_network,offline_network,exp_batch,gamma,device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_network.parameters(), 10)
                optimizer.step()
            if total_overall_steps%1000 == 0:
                offline_network.load_state_dict(online_network.state_dict())
            if truncated or terminated:
                break
        total_steps += step_count
        if (i + 1) % 100 == 0:
            current_avg = total_steps / 100
            loss_str = f"{loss.item():.4f}" if len(experiences) > 1000 else "N/A"
            if current_avg >= best_avg_reward:
                best_avg_reward = current_avg
                torch.save(online_network.state_dict(), save_path)
                print(f"--> New Best Model Saved! Avg Steps: {best_avg_reward:.1f}")
            print(f"Episode: {j + 1} | Avg Steps: {total_steps / 100:.1f} | Epsilon: {epsilon:.3f} | Loss: {loss.item():.4f}")
            total_steps = 0

        epsilon = max(epsilon * decay, epsilon_min)
    env.close()
else:
    model = A.agent()
    model.to(device)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    else:
        print("File not found, using fresh model.")
    env = gymnasium.make("LunarLander-v2", render_mode="human")
    state, _ = env.reset()
    done = False
    while not done:
        action = model.choice(torch.tensor(state, dtype=torch.float32).to(device), epsilon=0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()