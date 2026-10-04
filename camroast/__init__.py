"""CamRoast: two skeletons that comment on trick-or-treaters.

Submodules:
- app: camera loop, detection, overlay and trigger logic
- presence: time-based person confirmation and best-frame selection
- show: the asynchronous show (attention clip -> joke -> two voices)
- llm: joke generation with the OpenAI Responses API (structured JSON output)
- stt: speech-to-text for the talk-back mode
- tts: ElevenLabs synthesis and streaming playback
- voice: microphone capture, voice activity detection, utterance segmentation
- ui: overlay with buttons, status and Unicode subtitles
- premade: pre-recorded clips
"""
