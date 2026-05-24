"""Discord slash commands.

/start           — onboarding: 5–7 questions covering location, interests, schedule,
                   and vibe preference; answers written directly to mem0.
/profile         — mem0.get_all(user_id) → formatted readable summary sent as Discord reply.
/forget <topic>  — mem0 search + delete for facts matching topic; ack on completion.
/digest on|off   — toggle the APScheduler digest job; persists state to digest_config table.

Available from:
  /profile + /forget  → Week 3–4 (live when memory extraction goes live)
  /digest             → Week 5–6 (live when APScheduler digest is wired up)
  /start              → Week 9–10
"""
