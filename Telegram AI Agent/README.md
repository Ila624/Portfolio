# Telegram AI Agent with Function Calling

<img width="1713" height="610" alt="Screenshot (97)" src="https://github.com/user-attachments/assets/c7099076-9979-4821-8ed5-f2d92c633d26" />


## 🚀 Key Features & Architecture
Autonomous Tool Use: Powered by Groq for ultra-low latency inference, the central AI Agent dynamically decides when and how to call external tools (Google Calendar, Contact Management, and Email Services) based on user intent.

File Processing Pipeline: Features a dedicated logic branch to handle incoming files via Telegram. Files are parsed using custom JavaScript snippets and routed through optimized HTTP Requests directly to the LLM orchestration layer.

Contextual Memory: Implements conversational memory to maintain context and deliver coherent, multi-turn interactions.

## 🛠️ Tech Stack
Orchestration: n8n Workflows

LLM Engine: Groq API

Languages & Protocols: JavaScript, REST APIs, JSON

## Files included
Import this [file](./Telegram%20Agent.json) into your n8n instance to test or replicate the architecture
