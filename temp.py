import os
import random

# Create folders for training and test data
os.makedirs("lyrics", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Helper function to generate random lyrics
def generate_lyrics(num_lines=100, words_per_line=10):
    vocabulary = ["love", "hate", "night", "day", "fire", "rain", "sky", "dream", "heart", "soul",
                  "dance", "cry", "sing", "run", "fall", "light", "dark", "shine", "feel", "break"]
    return "\n".join(
        " ".join(random.choice(vocabulary) for _ in range(words_per_line))
        for _ in range(num_lines)
    )

# Generate training data
for i in range(50):
    valence = round(random.uniform(0, 1), 3)
    energy = round(random.uniform(0, 1), 3)
    lyrics = generate_lyrics(num_lines=200, words_per_line=10)
    with open(f"lyrics/song_{i:03}.txt", "w", encoding="utf-8") as f:
        f.write(f"{energy}\n{valence}\n{lyrics}")

# Generate test data
for i in range(10):
    lyrics = generate_lyrics(num_lines=200, words_per_line=10)
    with open(f"test/test_{i:03}.txt", "w", encoding="utf-8") as f:
        f.write(lyrics)

"Fake lyrics data generated for training (50 songs) and testing (10 songs)."
