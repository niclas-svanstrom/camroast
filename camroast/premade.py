# camroast/premade.py
"""Pre-recorded audio: premade joke pairs, attention clips and filler clips."""
import os
import random
import re

AUDIO_EXT = (".mp3", ".wav", ".ogg")


def _numeric_key(name: str):
    m = re.search(r"\d+", name)
    if m:
        return (0, int(m.group(0)))
    return (1, name.lower())


def load_premade_pairs(root: str):
    """Each subfolder holds one pair: a file with 'skalle' in the name and one with 'ben'."""
    pairs = []
    if not os.path.isdir(root):
        return pairs
    for sub in sorted(os.listdir(root), key=_numeric_key):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        skalle = ben = None
        for fn in sorted(os.listdir(d)):
            low = fn.lower()
            if not low.endswith(AUDIO_EXT):
                continue
            if "skalle" in low:
                skalle = os.path.join(d, fn)
            elif low.startswith("ben") or "benrangel" in low:
                ben = os.path.join(d, fn)
        if skalle and ben:
            pairs.append((skalle, ben))
    return pairs


def load_audio_files(root: str):
    files = []
    try:
        if os.path.isdir(root):
            for fn in os.listdir(root):
                if fn.lower().endswith(AUDIO_EXT):
                    files.append(os.path.join(root, fn))
    except Exception:
        pass
    return files


load_attention_files = load_audio_files  # backwards compatible name


def _ensure_mixer():
    import pygame
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    except Exception:
        try:
            pygame.mixer.quit()
            pygame.mixer.init()
        except Exception:
            pass


def play_premade_pair(paths):
    """Blocking: play the two files one after the other. Call from a worker thread."""
    import pygame
    _ensure_mixer()
    for p in paths:
        try:
            ch = pygame.mixer.Sound(p).play()
            clock = pygame.time.Clock()
            while ch is not None and ch.get_busy():
                clock.tick(20)
        except Exception as e:
            print(f"Failed to play {p}: {e}")


def play_random_attention(files):
    """Fire-and-forget: start one random clip. Returns the pygame channel (or None)."""
    import pygame
    if not files:
        return None
    _ensure_mixer()
    try:
        return pygame.mixer.Sound(random.choice(files)).play()
    except Exception as e:
        print(f"Failed to play attention sound: {e}")
        return None
