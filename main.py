from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
from data.questions import QUESTIONS
import random

def run_adaptive_assessment(questions):
    track = "web"
    env = AdaptiveAssessmentEnv(questions, track)
    agent = RLAssessmentAgent(env)
    
    while True:
        state = env.current_level
        action = agent.choose_action(state)
        agent.adjust_difficulty(action)
        
        question = env.get_question(state)
        if not question:
            print("No more questions available at this level.")
            break

        print(f"\nLevel: {env.current_level}")
        print(f"Question: {question['text']}")
        print(f"Options: {', '.join(question['options'])}")
        
        correct_answer = question['correct_answer']
        answer_probability = env.student_ability * 0.8 + 0.1

        if random.random() < answer_probability:
            answer = correct_answer
            print("Student answers correctly")
        else:
            answer = random.choice([a for a in question['options'] if a != correct_answer])
            print("Student answers incorrectly")
        
        reward, done = env.submit_answer(question, answer)
        next_state = env.current_level
        agent.update_q_table(state, action, reward, next_state)
        
        if done:
            break
    
    print("\nAssessment Complete!")
    correct = sum(1 for q in env.question_history if q['is_correct'])
    print(f"Final Score: {correct}/{len(env.question_history)}")
    print(f"Estimated Student Ability: {env.student_ability:.2f}")
    print("Question History:")
    for i, q in enumerate(env.question_history, 1):
        print(f"{i}. Level {q['level']}: {'Correct' if q['is_correct'] else 'Incorrect'}")

if __name__ == "__main__":
    run_adaptive_assessment(QUESTIONS)
