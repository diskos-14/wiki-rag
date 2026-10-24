import uuid

from app.conversations.repository import ConversationRepository


async def test_conversation_round_trip_and_recent_turns_order(session):
    repo = ConversationRepository(session)
    conversation_id = await repo.create()

    for i in range(4):
        await repo.add_message(
            conversation_id, role="user", content=f"q{i}", standalone_question=f"s{i}"
        )
        await repo.add_message(
            conversation_id, role="assistant", content=f"a{i}", source_chunk_ids=[i]
        )

    assert await repo.exists(conversation_id)
    assert not await repo.exists(uuid.uuid4())

    # The last 3 messages, oldest first.
    recent = await repo.recent_turns(conversation_id, limit=3)
    assert [(t.role, t.content) for t in recent] == [
        ("assistant", "a2"),
        ("user", "q3"),
        ("assistant", "a3"),
    ]

    messages = await repo.messages(conversation_id)
    assert len(messages) == 8
    assert messages[0].standalone_question == "s0"
    assert messages[1].source_chunk_ids == [0]
