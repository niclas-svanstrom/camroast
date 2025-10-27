# camroast/pipeline.py
import asyncio
from . import llm, tts

async def roast_once(frame, labels, voice_ids, v_enc_jpg):
    b64 = v_enc_jpg(frame)
    joke = await asyncio.to_thread(llm.generate_roast_from_image, b64)
    print(joke)
    lines = llm.assign_alternating_voices(joke)
    if not lines:
        return joke

    lines = [
        (voice_ids[0], lines[0][1]),
        (voice_ids[1], lines[1][1]),
    ]

    tasks = [asyncio.create_task(tts.synth_audio_async(txt, vce)) for vce, txt in lines]
    first_bytes = await tasks[0]
    tts.play_bytes(first_bytes)
    for t in tasks[1:]:
        tts.play_bytes(await t)
    return joke

