import os
import random


def load_premade_pairs(root: str):
    def numeric_key(name: str):
        import re as _re
        m = _re.search(r"\d+", name)
        if m:
            try:
                return (0, int(m.group(0)))
            except Exception:
                pass
        return (1, name.lower())

    pairs = []
    if not os.path.isdir(root):
        return pairs
    for sub in sorted(os.listdir(root), key=numeric_key):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        skalle = None
        ben = None
        for fn in sorted(os.listdir(d)):
            low = fn.lower()
            if any(low.endswith(ext) for ext in (".mp3", ".wav", ".ogg")):
                if low.startswith("skalle") or "skalle" in low:
                    skalle = os.path.join(d, fn)
                elif low.startswith("ben") or "benrangel" in low:
                    ben = os.path.join(d, fn)
        if skalle and ben:
            pairs.append((skalle, ben))
    return pairs


def load_attention_files(root: str):
    files = []
    try:
        if not os.path.isdir(root):
            return files
        for fn in os.listdir(root):
            low = fn.lower()
            if any(low.endswith(ext) for ext in (".mp3", ".wav", ".ogg")):
                files.append(os.path.join(root, fn))
    except Exception:
        pass
    return files


def _ensure_mixer():
    import pygame
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
    except Exception:
        # Try a safe default init if first attempt failed
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            pygame.mixer.init()
        except Exception:
            pass


def play_premade_pair(paths):
    import pygame
    _ensure_mixer()
    for p in paths:
        try:
            snd = pygame.mixer.Sound(p)
            ch = snd.play()
            clock = pygame.time.Clock()
            while ch.get_busy():
                clock.tick(10)
        except Exception as e:
            print(f"Failed to play {p}: {e}")


def play_random_attention(files: list[str]):
    # Fire-and-forget: start playing one random file, do not block
    import pygame
    if not files:
        return
    _ensure_mixer()
    try:
        p = random.choice(files)
        snd = pygame.mixer.Sound(p)
        snd.play()  # non-blocking
    except Exception as e:
        print(f"Failed to play attention sound: {e}")
