# FlashBack — Project Description (200 words)

**FlashBack** is a system that leverages NVIDIA Cosmos-Reason2 to solve a critical fire safety challenge: **tracing fire back to its origin** from surveillance footage. Unlike existing systems that only detect whether a fire exists, FlashBack analyzes the fire *spread process* over time to infer *where it started*.

Surveillance camera frame sequences are converted into temporal video, then analyzed using Cosmos-Reason2's physics reasoning capabilities — flame propagation patterns, smoke dispersion dynamics, and convection-driven spread. By tracing these patterns in reverse chronological order, the system estimates the origin as both a text description ("bottom-left of the greenhouse") and precise image coordinates (x=0.25, y=0.75).

**Core Innovation**: Origin coordinates are visualized directly on the original frames — a red crosshair marks the predicted origin, yellow arrows indicate spread direction. Lucas-Kanade optical flow tracks the origin across frames as the camera moves, producing a tracked demo video.

**Results**: Across 11 fire scenes (FLAME/SMOKE/NORMAL), the system achieved 100% origin tracing rate and 100% temporal reasoning rate.

**Physics Reasoning in Action**: The model explains ignition points through convection patterns, fuel density correlations, and temporal spread characteristics, presenting its analysis step-by-step via Chain-of-Thought reasoning.
