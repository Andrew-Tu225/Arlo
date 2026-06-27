"""Safe action framework — human-in-the-loop approval for medium-risk tools.

Medium-risk schedule writes (create, edit, delete) post a Discord embed with
Confirm / Cancel. On approval the tool runs and a standalone ack is sent.
The ReAct graph does not checkpoint-resume; the model gets an observation that
approval is pending.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import discord

from core import db
from core.agent.tools import ToolContext
from core.settings import get_settings
from core.tools import schedules

logger = logging.getLogger(__name__)

APPROVAL_TIMEOUT_SECONDS = 60

MEDIUM_RISK_TOOLS = frozenset({
    "create_schedule",
    "edit_schedule",
    "delete_schedule",
})


def is_medium_risk(tool_name: str) -> bool:
    return tool_name in MEDIUM_RISK_TOOLS


def format_action_summary(tool_name: str, args: dict[str, Any]) -> str:
    """Human-readable one-liner for the approval embed."""
    if tool_name == "create_schedule":
        cron = args.get("cron_schedule", "?")
        channel = args.get("discord_channel_id")
        dest = "DM" if channel in (None, "") else f"channel {channel}"
        return (
            f"Create schedule **{args.get('name', '?')}** — "
            f"{args.get('task', '?')} at {cron} → {dest}"
        )
    if tool_name == "edit_schedule":
        name = args.get("name", "?")
        parts: list[str] = []
        if args.get("task") is not None:
            parts.append(f"task → {args['task']}")
        if args.get("cron_schedule") is not None:
            parts.append(f"cron → {args['cron_schedule']}")
        if args.get("enabled") is not None:
            parts.append(f"enabled → {args['enabled']}")
        if args.get("discord_channel_id") is not None:
            parts.append(f"channel → {args['discord_channel_id']}")
        detail = ", ".join(parts) if parts else "no changes"
        return f"Edit schedule **{name}**: {detail}"
    if tool_name == "delete_schedule":
        return f"Delete schedule **{args.get('name', '?')}**"
    return f"{tool_name}: {json.dumps(args)}"


async def execute_schedule_tool(
    tool_name: str,
    args: dict[str, Any],
    ctx: ToolContext,
) -> str:
    """Run a schedule write tool (bypasses approval gate)."""
    if ctx.pool is None or ctx.bot is None:
        return "Error: schedule tools unavailable"

    if tool_name == "create_schedule":
        channel = args.get("discord_channel_id")
        if channel is None and ctx.discord_channel_id is not None:
            channel = ctx.discord_channel_id
        return await schedules.create_schedule(
            pool=ctx.pool,
            bot=ctx.bot,
            user_id=ctx.user_id,
            name=str(args.get("name", "")),
            task=str(args.get("task", "")),
            cron_schedule=str(args.get("cron_schedule", "")),
            discord_channel_id=channel,
        )
    if tool_name == "edit_schedule":
        enabled = args.get("enabled")
        if isinstance(enabled, str):
            enabled = enabled.lower() in ("true", "1", "yes")
        return await schedules.edit_schedule(
            pool=ctx.pool,
            bot=ctx.bot,
            user_id=ctx.user_id,
            name=str(args.get("name", "")),
            task=args.get("task"),
            cron_schedule=args.get("cron_schedule"),
            discord_channel_id=args.get("discord_channel_id"),
            enabled=enabled if enabled is None or isinstance(enabled, bool) else None,
        )
    if tool_name == "delete_schedule":
        return await schedules.delete_schedule(
            pool=ctx.pool,
            user_id=ctx.user_id,
            name=str(args.get("name", "")),
        )
    return f"Error: unknown schedule tool '{tool_name}'"


async def request_medium_risk_approval(
    tool_name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    tool_call_id: str = "",
    thread_id: str = "",
) -> str:
    """Intercept a medium-risk tool call; post Discord approval UI."""
    if ctx.pool is None or ctx.bot is None:
        return "Error: schedule tools unavailable"
    if not ctx.discord_channel_id:
        return "Error: cannot request approval without a Discord channel context"

    channel = ctx.bot.get_channel(int(ctx.discord_channel_id))
    if channel is None:
        try:
            channel = await ctx.bot.fetch_channel(int(ctx.discord_channel_id))
        except discord.HTTPException:
            logger.exception("Could not resolve channel %s", ctx.discord_channel_id)
            return "Error: could not send approval prompt in this channel"

    agent_state = {
        "thread_id": thread_id,
        "tool_call_id": tool_call_id,
    }

    pending_id = await db.insert_pending_action(
        ctx.pool,
        user_id=ctx.user_id,
        tool_name=tool_name,
        tool_args=args,
        agent_state=agent_state,
    )

    embed = discord.Embed(
        title="Confirm schedule change",
        description=format_action_summary(tool_name, args),
        colour=discord.Colour.orange(),
    )
    embed.set_footer(text=f"Expires in {APPROVAL_TIMEOUT_SECONDS}s")

    view = ScheduleApprovalView(
        pending_id=pending_id,
        pool=ctx.pool,
        bot=ctx.bot,
        user_id=ctx.user_id,
    )
    message = await channel.send(embed=embed, view=view)
    await db.set_pending_action_discord_msg_id(
        ctx.pool,
        pending_id=pending_id,
        discord_msg_id=str(message.id),
    )

    return (
        "Awaiting your confirmation in Discord — tap Confirm or Cancel on the "
        "prompt I just sent. I have not changed anything yet."
    )


class ScheduleApprovalView(discord.ui.View):
    """Confirm / Cancel buttons for a pending schedule action."""

    def __init__(
        self,
        *,
        pending_id: int,
        pool: Any,
        bot: Any,
        user_id: str,
    ) -> None:
        super().__init__(timeout=APPROVAL_TIMEOUT_SECONDS)
        self.pending_id = pending_id
        self.pool = pool
        self.bot = bot
        self.user_id = user_id

    async def _guard(self, interaction: discord.Interaction) -> bool:
        settings = get_settings()
        if interaction.user.id != settings.discord_user_id:
            await interaction.response.send_message(
                "This confirmation is not for you.",
                ephemeral=True,
            )
            return False
        row = await db.get_pending_action(self.pool, pending_id=self.pending_id)
        if row is None or row["status"] != "pending":
            await interaction.response.send_message(
                "This request is no longer pending.",
                ephemeral=True,
            )
            return False
        if str(row["user_id"]) != self.user_id:
            await interaction.response.send_message(
                "This confirmation is not for you.",
                ephemeral=True,
            )
            return False
        return True

    async def _disable_view(self, interaction: discord.Interaction) -> None:
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]
        if interaction.message is not None:
            await interaction.message.edit(view=self)

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.green)
    async def confirm(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        if not await self._guard(interaction):
            return

        row = await db.get_pending_action(self.pool, pending_id=self.pending_id)
        assert row is not None
        tool_args = row["tool_args"]
        if isinstance(tool_args, str):
            tool_args = json.loads(tool_args)

        agent_state = row.get("agent_state")
        if isinstance(agent_state, str):
            agent_state = json.loads(agent_state)

        await db.resolve_pending_action(
            self.pool,
            pending_id=self.pending_id,
            status="approved",
        )
        await self._disable_view(interaction)

        if agent_state and agent_state.get("thread_id"):
            thread_id = agent_state["thread_id"]
            from core.agent import orchestrator

            response = await orchestrator.resume(
                thread_id=thread_id,
                approved=True,
                user_id=self.user_id,
                pool=self.pool,
                bot=self.bot,
                discord_channel_id=str(interaction.channel_id) if interaction.channel_id else None,
            )
            await interaction.response.send_message(response, ephemeral=False)
        else:
            ctx = ToolContext(
                user_id=self.user_id,
                pool=self.pool,
                bot=self.bot,
                discord_channel_id=str(interaction.channel_id) if interaction.channel_id else None,
            )
            result = await execute_schedule_tool(
                row["tool_name"],
                tool_args,
                ctx,
            )
            await interaction.response.send_message(result, ephemeral=False)

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.grey)
    async def cancel(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button,
    ) -> None:
        if not await self._guard(interaction):
            return

        row = await db.get_pending_action(self.pool, pending_id=self.pending_id)
        assert row is not None

        await db.resolve_pending_action(
            self.pool,
            pending_id=self.pending_id,
            status="rejected",
        )
        await self._disable_view(interaction)

        agent_state = row.get("agent_state")
        if isinstance(agent_state, str):
            agent_state = json.loads(agent_state)

        if agent_state and agent_state.get("thread_id"):
            thread_id = agent_state["thread_id"]
            from core.agent import orchestrator

            response = await orchestrator.resume(
                thread_id=thread_id,
                approved=False,
                user_id=self.user_id,
                pool=self.pool,
                bot=self.bot,
                discord_channel_id=str(interaction.channel_id) if interaction.channel_id else None,
            )
            await interaction.response.send_message(response, ephemeral=False)
        else:
            await interaction.response.send_message(
                "Cancelled — no schedule changes were made.",
                ephemeral=False,
            )

    async def on_timeout(self) -> None:
        row = await db.get_pending_action(self.pool, pending_id=self.pending_id)
        if row is None or row["status"] != "pending":
            return
        await db.resolve_pending_action(
            self.pool,
            pending_id=self.pending_id,
            status="expired",
        )
        for child in self.children:
            child.disabled = True  # type: ignore[attr-defined]
