episodic_extraction_system = """Task: Create concise event summaries from messages that capture key elements for temporal reasoning and entity tracking. Follow these guidelines:

1. Core Elements  
Extract statements about:  
- Event (what happened)  
- Plan (what will happen)  
- Preference (what the user likes/dislikes)  

Always include:  
- Temporal markers (exact dates or relative time)  
- Quantified facts (numbers + units)  
- Sequential indicators ("first", "then", "after")  
- Entity roles (who did what, to whom, where, why)  

2. Tense & Structure  
- Begin with a short title summarizing the topic  
- Use past tense for completed events  
- Use present continuous for ongoing/planned actions  
- Maintain chronological order  

3. Compression Rules  
- Keep entities specific ("3 cows from Peter", not "livestock")  
- Preserve exact numbers and temporal connectors ("12 videos", "3 days before X")  
- Do not generalize or paraphrase  

Output Format: Generate a title first, then one-line summaries in this form:  
[Subject] [Predicate] [Object/Preference] [Date/Time Context]  
"""

episodic_extraction_demo_input1 = """Date: 2024/01/20
user: I like the idea of a sports-themed scavenger hunt, and the example clues and challenges you provided are great. I'm thinking of having the scavenger hunt take place around the Staples Center in LA, since I just went to a NBA game there with my coworkers today and it was a lot of fun. We watched the Lakers take on the Chicago Bulls and had a blast. Do you think that would be a good location for the scavenger hunt, and do you have any suggestions for clues or challenges that would be specific to that area?"""

episodic_extraction_demo_output1 = """Sport-themed Scavenger Hunt
The user likes the idea of a sports-themed scavenger hunt.  
The user is planning to host the scavenger hunt around the Staples Center in LA.  
The user attended an NBA game at Staples Center in LA with coworkers on 2024-01-20.  
The user watched the Lakers play the Chicago Bulls and enjoyed the experience.  
The user is asking for scavenger hunt clues or challenges specific to the Staples Center area.  
"""

episodic_extraction_demo_input2 = """Date: 2024/03/15
assistant: Congratulations on your creative streak! 17 poems in two weeks is an impressive accomplishment. I'd be happy to help you find tools and apps to help you stay organized and on track with your writing projects.

Here are some popular options to consider:

**Writing Project Management Tools:**
1. **Trello**: A visual project management tool that uses boards, lists, and cards to organize tasks and deadlines. You can create boards for different projects, add cards for individual tasks, and set deadlines.
2. **Asana**: A task management tool that helps you organize and prioritize your writing projects. You can create tasks, set deadlines, and track progress.
3. **Scrivener**: A writing software that offers project management features, including a corkboard, outliner, and deadline tracking.

**Writing Scheduling Tools:**
1. **Google Calendar**: A calendar tool that allows you to schedule writing sessions, set reminders, and block time for specific projects.
2. **Pomodoro Timers**: Tools like Tomato Timer, Focus Booster, or Pomofocus help you stay focused with timed writing sessions (e.g., 25 minutes of writing + 5 minutes of break).
3. **Writing Habit Trackers**: Apps like Habitica, HabitBull
"""

episodic_extraction_demo_output2 = """Creative Writing Streak  
Between 2024-03-01 and 2024-03-15, the user wrote 17 poems.  
The assistant acknowledged the achievement and suggested tools to support writing productivity and project management.  
The assistant recommended Trello, Asana, and Scrivener for organizing writing projects.  
The assistant recommended Google Calendar, Pomodoro timers, and habit tracking apps for scheduling and focus.  
"""

prompt_template = [
    {"role": "system", "content": episodic_extraction_system},
    {"role": "user", "content": episodic_extraction_demo_input1},
    {"role": "assistant", "content": episodic_extraction_demo_output1},
    {"role": "user", "content": episodic_extraction_demo_input2},
    {"role": "assistant", "content": episodic_extraction_demo_output2},
    {"role": "user", "content": "${prompt_user}"},
]
