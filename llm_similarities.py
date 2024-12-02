#!/usr/bin/env python3

import google.generativeai as genai
from gridworld_embedding_space import GridWorldEmbeddingSpace
import os
import matplotlib.pyplot as plt
from interactivePlotABCD import InteractivePlotABCD


def main():
    embedding_space = GridWorldEmbeddingSpace(TASK="gridworld", N_GRAM_TYPE="state-action", load_agents=True)
    trajectories = embedding_space.optimal_trajectories
    rewards = embedding_space.optimal_rewards 
    state_sequences, actions_sequences, rewards_sequences = traj_to_sentences(trajectories, rewards)
    
    s = [("Trajectory " + str(i) + " states:" + state_sequences[i]) for i in range(0, len(state_sequences))]
    s_str = ','.join(s)
    # print(s_str)
    
    a = [("Trajectory " + str(i) + " actions:" + actions_sequences[i]) for i in range(0, len(actions_sequences))]
    a_str = ','.join(a)
    
    r = [("Trajectory " + str(i) + " rewards:" + rewards_sequences[i]) for i in range(0, len(rewards_sequences))]
    r_str = ','.join(r)

    traj1 = 53
    traj2 = 59
    prompt = "Here are a series of state, action, and reward sequences for several trajectories. Give me the indices of two trajectories which share the same relationship with each other as trajectories " + str(traj1) + " and " + str(traj2) + "." 
    prompt += "These cannot be trajectories" + str(traj1) + " and " + str(traj2) + "."
    prompt += s_str + ", "
    prompt += a_str + ", "
    prompt += r_str
    # ----------------------------------------------------------------
    
    genai.configure(api_key='AIzaSyDRxCLCKF95i2leizDmc5K23sxm4MWmqNc')

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(prompt)
    print(response.text)
    
    embedding_space.load_model()
    
    # ----------------------------------------------------------------
    InteractivePlotABCD(embedding_space)


def traj_to_sentences(optimal_trajectories, optimal_rewards):
    state_sequences = [] 
    actions_sequences = []
    rewards_sequences = []
    
    for traj in optimal_trajectories:
        # get the list of states for the current trajectory
        s_sequence = [("(" + str(s[0]) + "," + str(s[1]) + ")") for s in [t[0] for t in traj]]
        states = ','.join(s_sequence)
        
        a_sequence = [t[1] for t in traj]  
        actions =  ','.join(a_sequence)
        
        r_sequence = [str(r) for r in optimal_rewards]
        rewards = ','.join(r_sequence)
        # '.'.join(rewards)
        # print(rewards)
        state_sequences.append(states)
        actions_sequences.append(actions)   
        rewards_sequences.append(rewards)

    return state_sequences, actions_sequences, rewards_sequences
        

if __name__ == "__main__":
    main()