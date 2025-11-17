def run_episode(env, agents, lm):
    env.reset();
    
    for round in range(env.max_rounds):
        print(f"c1: {env.c1}")
        print(f"c2: {env.c2}")
        print(f"\n--- ROUND: {round+1} ---")

        for proposer in agents:
            #proposal is made by whichever agents turn it is
            side, offset, crib = proposer.propose()
            implied = env.implied_fragment(side, offset, crib)
            print(f"{proposer.name} proposes {crib} at {offset} on P{side}")
            print(f"Implied fragment:{implied}")

            #agents vote
            yes_votes = 0
            for a in agents:
                vote = a.vote(implied, -25)
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


