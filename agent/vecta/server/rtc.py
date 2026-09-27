"""One WebRTC peer connection per phone: video track in, control channel both ways.

Signalling is a single HTTP offer/answer exchange (see `vecta.server.app`); on
the tailnet both ends have stable addresses, so no STUN/TURN is configured.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from aiortc import RTCConfiguration, RTCDataChannel, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from vecta.protocol import messages as m
from vecta.server.sessions import Session

log = logging.getLogger(__name__)

# Called with the decoded message; returns replies to send back (possibly none).
Handler = Callable[["Peer", m.Message], Awaitable[list[m.Message]]]


class Peer:
    """Wraps an RTCPeerConnection plus the session it feeds."""

    def __init__(
        self, session: Session, handler: Handler, on_close: Callable[[Peer], None] | None = None
    ) -> None:
        self.session = session
        self._on_close = on_close
        # No STUN/TURN: both ends are on the tailnet with stable addresses, and the
        # default STUN lookups add ~5 s to gathering on multi-interface hosts.
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.channel: RTCDataChannel | None = None
        self._handler = handler
        self._tasks: set[asyncio.Task[None]] = set()

        self.pc.on("track", self._on_track)
        self.pc.on("datachannel", self._on_datachannel)
        self.pc.on("connectionstatechange", self._on_state)

    async def answer(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        await self.pc.setRemoteDescription(offer)
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return self.pc.localDescription

    def send(self, msg: m.Message) -> None:
        if self.channel is None or self.channel.readyState != "open":
            log.warning("drop %s: channel not open", msg.type)
            return
        self.channel.send(m.encode(msg))

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.pc.close()
        if self._on_close is not None:
            self._on_close(self)

    # --- callbacks ---------------------------------------------------------------

    def _on_track(self, track: MediaStreamTrack) -> None:
        if track.kind != "video":
            return
        log.info("session %s: video track", self.session.id)
        task = asyncio.create_task(self._consume(track))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _consume(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                self.session.frames.push(await track.recv())
        except MediaStreamError:
            log.info("session %s: video track ended", self.session.id)

    def _on_datachannel(self, channel: RTCDataChannel) -> None:
        self.channel = channel
        channel.on("message", lambda data: self._spawn(self._on_message(data)))

    def _spawn(self, coro: Awaitable[None]) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _on_message(self, data: str | bytes) -> None:
        try:
            msg = m.decode(data)
        except m.ProtocolError as e:
            log.warning("session %s: bad message: %s", self.session.id, e)
            return
        for reply in await self._handler(self, msg):
            self.send(reply)

    async def _on_state(self) -> None:
        log.info("session %s: %s", self.session.id, self.pc.connectionState)
        if self.pc.connectionState in ("failed", "closed"):
            await self.close()
