import random
class RLAssessmentAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = {level: {'up': 1, 'stay': 1, 'down': 1} for level in env.levels}
        
    def choose_action(self, state=None):
        # Dynamic exploration based on performance
        exploration_rate = 0.3 * (1 - self.env.student_ability)
        
        if random.random() < exploration_rate:
            return random.choice(['up', 'stay', 'down'])
        
        # Prefer actions based on both Q-values and current performance
        if self.env.consecutive_correct >= 2:
            return 'up'
        elif self.env.consecutive_incorrect >= 2:
            return 'down'
        else:
            return 'stay'
        
    def update_q_table(self, state, action, reward, next_state, alpha=0.1, gamma=0.9):
        best_next = max(self.q_table[next_state].values())
        current = self.q_table[state][action]
        self.q_table[state][action] = current + alpha * (reward + gamma * best_next - current)

    def adjust_difficulty(self, action):
        current_idx = self.env.levels.index(self.env.current_level)
        
        if action == 'up' and current_idx < len(self.env.levels) - 1:
            new_idx = min(current_idx + 1, len(self.env.levels) - 1)
        elif action == 'down' and current_idx > 0:
            new_idx = max(current_idx - 1, 0)
        else:
            new_idx = current_idx
            
        self.env.current_level = self.env.levels[new_idx]
        
        # Update Q-table
        reward = 1 if action == ('up' if self.env.student_ability > 0.6 else 'stay') else -1
        self.q_table[self.env.current_level][action] += 0.1 * reward