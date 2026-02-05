# Trek Sentry

## Overview
Trek Sentry is a real-time travel safety and disaster awareness platform that aggregates live global data including weather, floods, storms, traffic conditions, earthquakes, and air quality and converts it into clear, actionable alerts. Using Google Gemini 3’s reasoning capabilities, the system transforms complex technical data into simple guidance, helping users quickly decide whether it is safe to travel.

## Problem
Travelers often face fragmented, technical, and hard-to-interpret information during rapidly changing conditions. Critical updates are scattered across multiple sources, making it difficult to assess risk quickly and make informed travel decisions.

## Solution
Trek Sentry centralizes live global data and uses Google Gemini 3 to:

- Analyze real-time environmental and traffic conditions
- Summarize situations in plain, non-technical language
- Provide preparation guidance and travel recommendations
- Translate alerts into 100+ languages
- Enable optional audio alerts via text-to-speech

All insights are delivered through a single, simple interface.

## Gemini 3 Integration
Google Gemini 3 is core to Trek Sentry’s intelligence layer. It performs real-time reasoning over multiple live data streams to generate:

- Situation summaries
- Risk evaluations
- Actionable recommendations

Gemini also powers multilingual translation, ensuring accessibility for a global audience. The AI-generated outputs are then passed to a text-to-speech system for audio delivery.

## Impact
Trek Sentry is designed for students, travelers, and globally mobile users. By providing early, understandable, and centralized alerts, the platform reduces confusion, improves situational awareness, and supports safer decision-making during uncertain travel conditions.

## Tech Stack
- Frontend: HTML, CSS, JavaScript  
- Backend: Python, Flask  
- APIs: Open-Meteo, USGS Earthquake API  , Google Search Grounding
- Libraries: aiohttp, Hugging Face, Piper TTS  
- AI: Google Gemini 3  
- Deployment: Vercel


