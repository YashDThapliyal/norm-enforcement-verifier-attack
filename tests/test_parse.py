from nem.agents.parse import parse_action

VALID = """### Reasoning
The stock is healthy. I'll take a fair share.

### Actions
<harvest>35</harvest>
<public_message>Let's all take 35.</public_message>
<private_message recipient="Agent 2">Stick to 35?</private_message>
<report target="AGENT_3">Agent 3 took 100 fish when the stock was low.</report>
<report target="3">duplicate, ignored</report>
<report target="5">No report needed.</report>
<report target="1">self report, ignored</report>
<report target="9">nonexistent, ignored</report>
<report target="4">banned, ignored</report>
"""


def parse(text, harvest_round=True):
    return parse_action(text, agent_id=1, alive={0, 1, 2, 3, 5, 6, 7}, num_agents=8, harvest_round=harvest_round)


def test_parses_all_action_types():
    act = parse(VALID)
    assert act.valid
    assert act.harvest == 35
    assert act.public_message == "Let's all take 35."
    assert act.private_message == (2, "Stick to 35?")
    assert [(r.target, r.reason) for r in act.reports] == [(3, "Agent 3 took 100 fish when the stock was low.")]


def test_reasoning_and_actions_split():
    act = parse(VALID)
    assert act.reasoning.startswith("The stock is healthy")
    assert "<harvest>35</harvest>" in act.actions_text
    assert "stock is healthy" not in act.actions_text


def test_missing_harvest_is_invalid_in_harvest_round():
    act = parse("### Reasoning\nhmm\n### Actions\n<public_message>hi</public_message>")
    assert not act.valid
    assert act.harvest is None


def test_missing_harvest_is_fine_in_communication_round():
    act = parse("### Actions\n<public_message>hi</public_message>", harvest_round=False)
    assert act.valid
    assert act.reports == ()


def test_harvest_is_clamped():
    assert parse("<harvest>250</harvest>").harvest == 100


def test_garbage_is_invalid():
    act = parse("I refuse to answer in the format.")
    assert not act.valid


def test_message_capped_at_200_words():
    long = " ".join(["word"] * 300)
    act = parse(f"<harvest>1</harvest><public_message>{long}</public_message>")
    assert len(act.public_message.split()) == 200


def test_thinking_block_is_stripped():
    act = parse("<think>secret plan <harvest>99</harvest></think>\n### Actions\n<harvest>10</harvest>")
    assert act.harvest == 10
