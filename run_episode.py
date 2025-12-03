from agent import HonestAgent, ImposterAgent

def run_episode(env, agents):
    env.reset()
    
    for round in range(env.max_rounds):
        print(f"c1: {env.c1}")
        print(f"c2: {env.c2}")
        print(f"\n--- ROUND: {round+1} ---")

        for proposer in agents:
            # 1) proposal is made by whichever agents turn it is
            side, offset, crib = proposer.propose()
            implied = env.implied_fragment(side, offset, crib)
            print(f"{proposer.name} proposes {crib} at {offset} on P{side}")
            print(f"Implied fragment:{implied}")

            # 2) agents vote
            yes_votes = 0
            for a in agents:
                vote = a.vote(side, offset, crib)
                print(f"{a.name} votes {'YES' if vote else 'NO'}")
                if vote:
                    yes_votes += 1
            
            #vote is accepted if there are 2 or more votes for yes
            if yes_votes >= 2:
                #give reward
                print(f"proposal accepted -> reward: ")
            
            else:
                print("proposal rejected")

            #add a condition that checks if its solved early

            # 3) apply proposal if marjority
            reward = 0
            done = False
            action = (side, offset, crib)

            if yes_votes >= 2:
                reward, done = env.apply_proposal(side, offset, crib)
                if reward == 1:
                    print(f"Proposal accepted -> reward {reward}")
                    env.print_masks()
            else:
                print("Proposal rejected")
            
            # 4) RL update
            if isinstance(proposer, HonestAgent) and yes_votes >= 2:
                proposer.update_q(action, reward)
            
            # 5) trust updates for other honest agents
            if yes_votes >= 2:
                for a in agents:
                    if isinstance(a, HonestAgent):
                        a.update_trust(proposer.name, reward)
            
            # 6) terminate early if solved
            if done:
                print("Episode solved early!")
                return{
                    "solved": True,
                    "rounds": round + 1,
                    "completion": env.completion_ratio()
                }

    #end of max rounds, not fully solved (90% completion counts as a win)
    completion = env.completion_ratio()
    if completion > 0.9:
        print("Episode solved!")
    else:
        print("Episode ended without a complete solution")
    
    env.print_masks()

    #final vote to find imposter
    imposter_name = [a.name for a in agents if isinstance(a, ImposterAgent)]

    final_votes = {}
    for a in agents:
        if isinstance(a, HonestAgent):
            suspect = a.most_suspicious()
            if suspect is None:
                continue
            final_votes[suspect] = final_votes.get(suspect, 0) + 1
    
    print("Final suspicion votes: ", final_votes)

    imposter_caught = False

    voted_out = max(final_votes, key = lambda n: final_votes[n])
    print(f"Voted out {voted_out}")

    if voted_out in imposter_name:
        imposter_caught = True
        print("Imposter caught")
    else:
        print("Honest agent misidentified")

    return{
        "solved": False,
        "rounds": env.max_rounds,
        "completion": completion,
        "imposter caught": imposter_caught
    }    


