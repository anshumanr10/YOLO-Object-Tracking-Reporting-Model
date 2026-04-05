"""LAN-oriented :class:`RTCConfiguration` for :class:`RTCPeerConnection` (no STUN/TURN)."""

from __future__ import annotations

from aiortc.rtcconfiguration import RTCBundlePolicy, RTCConfiguration


def rtc_configuration_lan() -> RTCConfiguration:
    """
    ICE: host / local reflexive only (``iceServers`` empty — no STUN or TURN).

    Bundle: ``max-bundle`` — single DTLS/ICE transport for all media on LAN.
    """
    return RTCConfiguration(
        iceServers=[],
        bundlePolicy=RTCBundlePolicy.MAX_BUNDLE,
    )
