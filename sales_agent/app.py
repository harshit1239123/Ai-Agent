import logging
import os

import chainlit as cl
import dotenv
from agents import InputGuardrailTripwireTriggered, Runner, SQLiteSession
from chainlit.server import app as fastapi_app
from fastapi.responses import PlainTextResponse
from openai.types.responses import ResponseTextDeltaEvent
from starlette.routing import Route

from sales_agent import build_sales_agent, create_exa_search_mcp

dotenv.load_dotenv()

logger = logging.getLogger(__name__)


async def health(_request) -> PlainTextResponse:
    # Unauthenticated, no LLM/DB calls — safe and free for an external cron/uptime
    # ping to hit every ~10 minutes to stop Render's free-tier instance from
    # sleeping after 15 minutes of inactivity.
    # https://<your-render-service>.onrender.com/health
    return PlainTextResponse("ok")


# Chainlit registers a catch-all "/{path:path}" route (to serve its SPA) at import
# time, and Starlette matches routes in registration order. Appending "/health"
# normally would never be reached, so it's inserted at the front of the route
# table instead.
fastapi_app.router.routes.insert(0, Route("/health", health, methods=["GET"]))


@cl.on_chat_start
async def on_chat_start():
    # Same memory pattern as chatbot_complete/3_memory.py and 4_authentication.py,
    # but keyed by Chainlit's per-chat session id instead of a fixed string, so
    # concurrent visitors each get their own conversation history instead of
    # sharing one SQLiteSession.
    session = SQLiteSession(cl.context.session.id)
    cl.user_session.set("agent_session", session)

    # Each chat session gets its own Exa MCP connection (or none, if EXA_API_KEY
    # isn't set / the connection fails) instead of sharing one global connection
    # across every visitor.
    exa_mcp = create_exa_search_mcp()
    if exa_mcp is not None:
        try:
            await exa_mcp.connect()
        except Exception:
            exa_mcp = None

    cl.user_session.set("exa_mcp", exa_mcp)
    cl.user_session.set("agent", build_sales_agent(exa_mcp))


@cl.on_chat_end
async def on_chat_end():
    exa_mcp = cl.user_session.get("exa_mcp")
    if exa_mcp is not None:
        await exa_mcp.cleanup()


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")
    agent = cl.user_session.get("agent")

    try:
        result = Runner.run_streamed(
            agent,
            message.content,
            session=session,
        )

        msg = cl.Message(content="")
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                await msg.stream_token(token=event.data.delta)

            elif (
                event.type == "raw_response_event"
                and hasattr(event.data, "item")
                and hasattr(event.data.item, "type")
                and event.data.item.type == "function_call"
                and len(event.data.item.arguments) > 0
            ):
               
                # "course_qna_lookup_tool" or the raw search query.
                with cl.Step(name="Thinking...", type="tool"):
                    pass

        await msg.update()

    except InputGuardrailTripwireTriggered:
        await cl.Message(
            content=(
                "I'm the sales assistant for the AI Ultimate Course, so I can only help "
                "with questions about the course, AI learning, or enrollment. "
                "Feel free to ask me anything about that!"
            )
        ).send()

    except Exception:
        # Chainlit's on_message handler doesn't surface unhandled exceptions to the
        # UI (only logs them server-side), which turns real failures into silent
        # "no response" symptoms for the user. Log the full traceback for debugging
        # and always show the user something instead of leaving them hanging.
        logger.exception("Unhandled error while answering a message")
        await cl.Message(
            content="Something went wrong on my end answering that — please try again."
        ).send()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == (
        os.getenv("CHAINLIT_USERNAME"),
        os.getenv("CHAINLIT_PASSWORD"),
    ):
        return cl.User(
            identifier=username,
            metadata={"role": "visitor", "provider": "credentials"},
        )
    else:
        return None
