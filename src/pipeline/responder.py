"""
The "brain" slot: transcript in, reply text out.

In a real voice agent this is an LLM call (and the dominant latency term).
It's a Protocol so the pipeline is testable and runnable offline:
EchoResponder closes the loop without any API key; ClaudeResponder is the
real thing — a stateful, per-session conversation with Claude.

Production note the docs expand on: we make one blocking API call per turn
and speak the whole reply. Real agents STREAM tokens and cut the stream at
the first sentence boundary so TTS starts while the LLM is still writing —
that overlap is worth several hundred ms of turnaround.
"""

from __future__ import annotations

from typing import Protocol


class Responder(Protocol):
    def respond(self, transcript: str) -> str: ...


class EchoResponder:
    """Repeat the transcript back — proves the full loop end to end."""

    def respond(self, transcript: str) -> str:
        return f"You said: {transcript}"


# Both LLM responders speak through a TTS engine, so they share this style.
VOICE_SYSTEM_PROMPT = (
    "You are a voice assistant. Your replies are spoken aloud by a "
    "text-to-speech engine, so answer in 1-3 short conversational "
    "sentences of plain prose: no markdown, no lists, no code, no "
    "headings. Be direct and natural, like talking to a person."
)


class ClaudeResponder:
    """A real conversational brain via the Claude API.

    Stateful: keeps the conversation history for ITS session (so "what did I
    just say?" works) — which is why the factory builds one per connection,
    like the VAD, not one per process like the model engines.

    Key from ANTHROPIC_API_KEY (env or .env), or an `ant auth login` profile.
    """

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 300,
                 api_key: str = "", max_turns: int = 20):
        import anthropic  # lazy: only needed when this responder is selected

        # Empty key -> let the SDK resolve env vars / auth profiles itself.
        self.client = anthropic.Anthropic(api_key=api_key or None)
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._messages: list[dict] = []

    def respond(self, transcript: str) -> str:
        self._messages.append({"role": "user", "content": transcript})
        # Keep a bounded history window (a voice session can run for hours).
        self._messages = self._messages[-2 * self.max_turns:]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=VOICE_SYSTEM_PROMPT,
            messages=self._messages,
        )
        reply = "".join(b.text for b in response.content if b.type == "text").strip()
        self._messages.append({"role": "assistant", "content": reply})
        return reply


class OpenAIResponder:
    """Same brain slot, OpenAI's API — structurally identical to Claude's:
    per-session history, bounded window, voice-tuned system prompt. The point
    of the Responder seam is exactly that vendors are interchangeable here.

    Key from OPENAI_API_KEY, model from OPENAI_MODEL (env or .env).
    """

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 300,
                 api_key: str = "", max_turns: int = 20):
        import openai  # lazy: only needed when this responder is selected

        self.client = openai.OpenAI(api_key=api_key or None)
        self.model = model
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self._messages: list[dict] = []

    def respond(self, transcript: str) -> str:
        self._messages.append({"role": "user", "content": transcript})
        self._messages = self._messages[-2 * self.max_turns:]

        response = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": VOICE_SYSTEM_PROMPT}]
            + self._messages,
        )
        reply = (response.choices[0].message.content or "").strip()
        self._messages.append({"role": "assistant", "content": reply})
        return reply
