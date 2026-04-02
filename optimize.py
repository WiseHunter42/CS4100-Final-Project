import torch
import variables

def optimize():
    # if we don't have enough data, don't try to optimize
    if len(variables.memory) < variables.batch_size:
        return
    
    # get random sample of batches from memory
    transitions = variables.memory.sample(variables.batch_size)

    # Group batch into one giant Transition with lists for each field
    batch = variables.Transition(*zip(*transitions))

    # Determine non-final next states and mask for next state values, which is just a T/F vector that says whether or not a given next state is None
    non_final_next_state_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device=variables.device, dtype=torch.bool)
    non_final_next_states = torch.cat([x for x in batch.next_state if x is not None])

    # Turn state, action, reward into tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # get the q values for the current state using the policy net
    q_values_policy = variables.policy_net(state_batch).gather(1, action_batch)

    # get the next_state values; use the non-final-next-state mask to only update non_final states
    next_state_values = torch.zeros(variables.batch_size, device=variables.device)
    with torch.no_grad():
        # max_a Q(s', a')
        next_state_values[non_final_next_state_mask] = variables.target_net(non_final_next_states).max(1).values # for each row of q values (for each state), get the best value
    
    q_values_target = reward_batch + (variables.gamma * next_state_values)

    # Compute Huber loss
    loss = variables.criterion(q_values_policy, q_values_target.unsqueeze(1))

    # Optimize the model
    variables.optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(variables.policy_net.parameters(), 10) # change clipping as needed
    variables.optimizer.step()
    variables.loss_history.append(loss.item())



