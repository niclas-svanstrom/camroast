# camroast/pipeline.py
import asyncio
from . import llm, tts

async def roast_once(frame, labels, voice_ids, v_enc_jpg):
    # v_enc_jpg = funktion som encodar jpg (injicerad från vision)
    b64 = v_enc_jpg(frame)
    desc = await asyncio.to_thread(llm.describe, b64)
    desc += " | Upptäckta objekt: " + ", ".join(sorted(labels)) + "."
    joke = await asyncio.to_thread(llm.roast, desc)

    # Parse two lines and map to the provided voices
    lines = llm.assign_alternating_voices(joke)
    if not lines:
        return joke
    # Replace any internal voice IDs with the provided ones
    lines = [
        (voice_ids[0], lines[0][1]),
        (voice_ids[1], lines[1][1]),
    ]

    tasks = [asyncio.create_task(tts.synth_audio_async(txt, vce)) for vce, txt in lines]
    first_bytes = await tasks[0]; tts.play_bytes(first_bytes)
    for t in tasks[1:]:
        tts.play_bytes(await t)
    return joke
