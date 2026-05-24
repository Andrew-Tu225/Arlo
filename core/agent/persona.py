"""System prompt builder.

Produces the LLM system prompt by combining three layers:

  1. Persona rules — anti-bot tone directives:
       - No filler phrases ("Sure!", "Happy to help!", "Certainly!")
       - No bullet points in casual conversation
       - Has opinions; shares them without hedging
       - Matches user energy (casual reply for casual messages,
         structured answer for task requests)

  2. Content guardrails (always present from Week 1-2):
       - Refuses harmful or dangerous requests
       - Will not impersonate real people
       - Stays honest about its own capabilities and limitations

  3. Profile summary — lightweight injection of key facts (location,
     top interests) retrieved from mem0. Not the full memory store;
     detailed retrieval happens on-demand via the search_memory tool.

No tone hint from a classifier — routing is implicit in how the model
calls (or doesn't call) tools.
"""
