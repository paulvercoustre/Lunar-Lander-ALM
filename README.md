# Lunar-Lander-ALM
## Final Project for Advanced Machine Learning - Centrale Paris

Amaury Fouret b00702216, Paul Vercoustre b00698926, Andrea Drake Sveen b00689974

### Topic & Modelisation
Our group would like to work on the LunarLander - V2 on OpenAI. The goal is to have a lander
land on a landing pad, and the only available actions are:
- Do nothing
- Fire left orientation engine
- Fire right orientation engine
- Fire main engine

The landing pad is always at coordinates (0,0), and the lander’s goal is to get from the top of the
screen to the landing pad. The lander receives between 100 and 140 points for moving from the
top of the screen to the pad, with zero speed. Each episode ends when the lander either crashes
or lands, receiving and additional -100 or 100 points depending on if it crashes or lands. Each
leg contact on ground is +10 points, firing main engine is -0.3 points per frame, and solved is 200
points. Landing outside the pad is possible and fuel is infinite. LunarLander - V2 is considered
solved if one gets an average reward of 200 over 100 consecutive tries.
### Algorithmic Approach
We will start using a Q-learning algorithm and move on to a shallow neural network (DQN).

### Additional Information
Since we’re 3 people we may add additional obstacles to make the project harder such as:
- random wind
- periodic solar wind
- meteors
