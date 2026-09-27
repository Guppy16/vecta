import pytest

from vecta.protocol import messages as m


def test_round_trip_every_type() -> None:
    samples = [
        m.SessionStart(session_id="abc", device={"w": 1, "h": 2}),
        m.TaskSet(text="find bread"),
        m.InputText(text="sort by protein"),
        m.UiEvent(event="tap", target="row-3", data={"x": 1}),
        m.Capture(kind="photo", id="c1", ts=123),
        m.Ping(t=5),
        m.SessionState(session_id="abc", task=None),
        m.Status(phase="thinking", text=""),
        m.PageRender(html="<h1>hi</h1>", version=3),
        m.CaptureRequest(kind="video", hint="pan left to right"),
        m.CaptureAck(id="c1", frames=12, url="http://x/y.jpg"),
        m.Pong(t=5, server_t=9),
    ]
    for msg in samples:
        assert m.decode(m.encode(msg)) == msg


def test_unknown_fields_are_ignored() -> None:
    assert m.decode('{"type":"ping","t":1,"future_field":true}') == m.Ping(t=1)


@pytest.mark.parametrize("text", ["not json", "[]", '{"no":"type"}', '{"type":"nope"}'])
def test_bad_messages_raise(text: str) -> None:
    with pytest.raises(m.ProtocolError):
        m.decode(text)
