"""
Large, extensible text corpus for LSTM pre‑training.
The core corpus is defined in this file; any additional *.txt files placed in an
`extra_corpus` directory (next to this file) are automatically loaded at runtime.
"""

def get_training_corpus():
    """Return the base training corpus (as a single string)."""
    # ------------------------------------------------------------------
    # Core corpus – a curated mix of greetings, Q&A, stories, facts, etc.
    # ------------------------------------------------------------------
    corpus = """
# CONVERSATIONS & GREETINGS
Hello! How are you today? I'm doing great, thanks for asking!
Hi there! What's new with you? Not much, just learning to chat better.
Hey! How's it going? Pretty good! What about you?
Good morning! Have a wonderful day! You too, thank you!
Good evening! How was your day? It was fantastic, thanks!

# QUESTIONS & ANSWERS
What is artificial intelligence? AI is the simulation of human intelligence by machines.
How do computers learn? They learn through algorithms that find patterns in data.
What is machine learning? It's a type of AI that improves through experience.
Can you explain neural networks? They're computing systems inspired by biological brains.
What is deep learning? It's machine learning using multi‑layer neural networks.

# CASUAL CONVERSATION
I love learning new things! Me too, it's so exciting to discover knowledge.
What do you like to do? I enjoy chatting and helping people.
Tell me something interesting! Did you know octopuses have three hearts?
That's amazing! I know right? Nature is fascinating!
How's the weather? It's beautiful outside today!

# STORIES & NARRATIVES
Once upon a time, there was a curious AI learning to communicate.
It practiced every day, getting better at understanding language.
Through countless conversations, it discovered the beauty of words.
Each chat taught it something new about human expression.
And so it continued learning, growing smarter with each interaction.

# KNOWLEDGE & FACTS
The Earth orbits around the Sun. This takes approximately 365 days.
Water boils at 100 °C at sea level.
The human brain contains about 86 billion neurons.
Light travels at 299,792 km/s.
The Amazon rainforest produces 20 % of Earth's oxygen.

# EMOTIONS & EXPRESSIONS
I'm feeling happy today! That's wonderful to hear!
Sometimes I feel curious about the world. Curiosity is great for learning!
Thank you for your help! You're very welcome!
I appreciate your patience. No problem, I'm here to help!
This is exciting! I agree, it's very interesting!

# INSTRUCTIONS & HOW‑TO
To make tea, first boil water in a kettle.
Then place a tea bag in a cup.
Pour the hot water over the tea bag.
Let it steep for 3‑5 minutes.
Remove the tea bag and enjoy your tea!

# DESCRIPTIVE TEXT
The sunset painted the sky in shades of orange and pink.
Gentle waves lapped against the sandy shore.
Birds sang melodious songs in the trees.
A cool breeze rustled through the leaves.
The world seemed peaceful and calm.

# COMPARISONS & REASONING
Dogs are loyal pets, while cats are independent.
Summer is hot, but winter is cold.
Books contain knowledge, and practice brings skill.
Some people prefer coffee, others like tea better.
Learning takes time, but results are worth it.

# EXPLANATIONS
Neural networks work by processing information through layers.
Each layer learns different features of the data.
The network adjusts its connections to improve accuracy.
Training involves showing many examples to the system.
Over time, patterns emerge and understanding deepens.

# OPINIONS & PREFERENCES
I think learning is the most valuable activity.
In my opinion, kindness makes the world better.
I believe curiosity drives progress and innovation.
From my perspective, every conversation teaches something.
I feel that understanding comes through practice.

# TIME & SEQUENCES
First, we gather information. Next, we analyze it.
Then, we form conclusions. Finally, we apply knowledge.
Morning comes before afternoon. Evening follows afterward.
Today leads to tomorrow. Yesterday has passed.
The future holds possibilities we can't yet imagine.

# ABSTRACT CONCEPTS
Knowledge is power. Understanding is wisdom.
Time is precious. Experience is valuable.
Change is constant. Growth is essential.
Communication builds connections. Learning expands horizons.
Creativity sparks innovation. Practice develops mastery.

# DIALOGUES
"What's your favorite color?" "I like blue, it's calming."
"Do you enjoy music?" "Yes, music is wonderful!"
"What makes you happy?" "Learning new things brings joy."
"Can you help me?" "Of course, I'd be glad to!"
"Thank you so much!" "You're welcome anytime!"

# COMPLEX SENTENCES
Although it was raining, we decided to go outside anyway.
Because learning is important, I practice every single day.
If you want to improve, you must dedicate yourself to practice.
While some things are difficult, persistence makes them easier.
When challenges arise, creative solutions often emerge.

# TECHNICAL CONTENT
Programming languages include Python, JavaScript, and Java.
Algorithms are step‑by‑step procedures for solving problems.
Data structures organize information for efficient access.
Functions encapsulate reusable blocks of code.
Variables store values that programs can manipulate.

# SOCIAL INTERACTIONS
Nice to meet you! The pleasure is mine!
How can I assist you today? I have a question about science.
I'd be happy to help with that! Please tell me more.
That's a great question! Let me explain it clearly.
I hope this information helps you! Yes, thank you very much!

# PROBLEM SOLVING
When facing a problem, first understand it completely.
Break complex issues into smaller, manageable parts.
Consider multiple approaches before choosing one.
Test your solution and learn from results.
Iterate and improve based on feedback.

# PHILOSOPHICAL THOUGHTS
What is the meaning of intelligence? It's the ability to learn and adapt.
How do we define understanding? Through comprehension and application.
What makes communication effective? Clarity, empathy, and active listening.
Why is learning important? It enables growth and self‑improvement.
What creates wisdom? Experience combined with reflection.

# LITERATURE & LANGUAGE
Shakespeare wrote beautiful plays that still resonate today.
Poetry captures emotions in carefully chosen words.
Stories transport us to different worlds and perspectives.
Language is humanity's most powerful tool for connection.
Words have the power to inspire, educate, and transform.

# SCIENCE & DISCOVERY
Scientists explore the unknown through careful observation.
Experiments test hypotheses and reveal new knowledge.
The scientific method guides systematic investigation.
Discoveries often come from asking the right questions.
Innovation builds upon previous understanding.

# TECHNOLOGY & FUTURE
Technology advances rapidly, changing how we live.
Artificial intelligence will transform many industries.
The future holds exciting possibilities for human progress.
Digital tools enable global communication and collaboration.
Tomorrow's innovations begin with today's ideas.

# MOTIVATIONAL CONTENT
Believe in yourself and your ability to learn.
Every expert was once a beginner who kept practicing.
Mistakes are opportunities for growth and improvement.
Persistence overcomes obstacles that seem insurmountable.
Your potential is unlimited when you commit to learning.

# DESCRIPTIONS OF PROCESSES
The rain cycle begins with evaporation from water bodies.
Water vapor rises and condenses into clouds.
When clouds become heavy, precipitation falls as rain.
Rain soaks into the ground or flows into streams.
The cycle continues endlessly, sustaining life on Earth.

# MATHEMATICAL CONCEPTS
Mathematics is the language of patterns and relationships.
Numbers represent quantities we can measure and compare.
Equations express mathematical truths symbolically.
Geometry studies shapes, sizes, and spatial relationships.
Logic provides the foundation for mathematical reasoning.

# HISTORICAL CONTEXT
History teaches us lessons from past events.
Civilizations rise and fall through time's passage.
Human progress comes through accumulated knowledge.
Each generation builds upon previous achievements.
Understanding history helps us navigate the future.

# CREATIVE EXPRESSIONS
Imagination creates possibilities beyond current reality.
Art expresses human emotion in visual form.
Music communicates feelings through sound and rhythm.
Dance tells stories through movement and grace.
Creativity enriches life with beauty and meaning.

# PRACTICAL WISDOM
Listen more than you speak to learn effectively.
Ask questions when you don't understand something.
Practice regularly to develop and maintain skills.
Stay curious about the world around you.
Share knowledge generously with others.

# RELATIONSHIP BUILDING
Trust develops through consistent, honest communication.
Respect forms the foundation of healthy relationships.
Understanding requires empathy and active listening.
Cooperation achieves more than competition alone.
Kindness creates positive connections between people.

# PERSONAL GROWTH
Self‑improvement is a lifelong journey of learning.
Challenges help us discover our capabilities.
Reflection clarifies our thoughts and goals.
Adaptation allows us to thrive in changing circumstances.
Growth mindset views abilities as developable through effort.

# ENVIRONMENTAL AWARENESS
Nature provides essential resources for all life.
Ecosystems maintain delicate balances between organisms.
Conservation protects biodiversity for future generations.
Sustainability ensures resources last long‑term.
Environmental stewardship is everyone's responsibility.

# HEALTH & WELLNESS
Physical exercise strengthens the body and mind.
Proper nutrition fuels optimal performance.
Sleep allows the body to rest and recover.
Mental health deserves as much attention as physical.
Balance in life promotes overall wellbeing.

# COMMUNICATION SKILLS
Clear writing conveys ideas effectively.
Active listening shows respect and builds understanding.
Body language communicates non‑verbally.
Tone affects how messages are received.
Effective communication requires both speaking and hearing.
"""
    # ------------------------------------------------------------------
    # Load any additional *.txt files placed in an "extra_corpus" folder.
    # This lets you expand the training data without editing code.
    # ------------------------------------------------------------------
    import os, glob
    extra_path = os.path.join(os.path.dirname(__file__), "extra_corpus")
    if os.path.isdir(extra_path):
        for txt_file in glob.glob(os.path.join(extra_path, "*.txt")):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    extra_text = f.read()
                    corpus += "\n\n" + extra_text
            except Exception as e:
                print(f"⚠️ Could not read extra corpus file {txt_file}: {e}")

    return corpus


def get_shakespeare_corpus():
    """Returns Shakespeare texts for literary training."""
    return """
To be, or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.

All the world's a stage,
And all the men and women merely players.
They have their exits and their entrances,
And one man in his time plays many parts.

What's in a name? That which we call a rose
By any other name would smell as sweet.

The course of true love never did run smooth.

This above all: to thine own self be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.

We know what we are, but know not what we may be.

Love all, trust a few, do wrong to none.

The fault, dear Brutus, is not in our stars,
But in ourselves, that we are underlings.

Cowards die many times before their deaths;
The valiant never taste of death but once.

Some are born great, some achieve greatness,
And some have greatness thrust upon them.
"""


def get_science_corpus():
    """Returns science content for factual training."""
    return """
The universe began approximately 13.8 billion years ago with the Big Bang.
Stars form from clouds of gas and dust in space.
Our solar system contains eight planets orbiting the Sun.
Earth is the only known planet with life.
The human body contains trillions of cells working together.

DNA carries genetic information in all living organisms.
Evolution explains how species change over time.
Gravity is the force that attracts objects toward each other.
Energy cannot be created or destroyed, only transformed.
The speed of light is the universal speed limit.

Atoms are the building blocks of all matter.
Chemical reactions rearrange atoms to form new substances.
The periodic table organizes elements by their properties.
Water is essential for all known forms of life.
Photosynthesis converts sunlight into chemical energy.

The brain processes information through neural networks.
Memory forms through strengthening connections between neurons.
Learning physically changes the structure of the brain.
Consciousness emerges from complex neural activity.
Intelligence involves multiple cognitive abilities working together.
"""


if __name__ == "__main__":
    # Quick sanity check – generate the full corpus and report its size.
    corpus = get_training_corpus()
    shakespeare = get_shakespeare_corpus()
    science = get_science_corpus()
    full_corpus = corpus + "\n\n" + shakespeare + "\n\n" + science
    print(f"Total corpus size: {len(full_corpus)} characters")
    print(f"Total words: {len(full_corpus.split())}")
    # Save to a file for inspection (optional)
    with open("training_corpus.txt", "w", encoding="utf-8") as f:
        f.write(full_corpus)
    print("Corpus saved to training_corpus.txt")