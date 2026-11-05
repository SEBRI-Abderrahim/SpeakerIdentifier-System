"""
====================================================================
SpeakerIdentity IA - PCM16 WebClient Audio Test
====================================================================
"""

import struct


def test_pcm16_mono_16khz():

    sample_rate = 16000
    channels = 1
    sample_width = 2

    duration = 1

    audio = b"\x00\x00" * (
        sample_rate
        * duration
    )

    assert len(audio) == (
        sample_rate
        * channels
        * sample_width
    )


def test_pcm16_alignment():

    audio = b"\x00\x00" * 1600

    assert len(audio) % 2 == 0


def test_pcm16_sample_decoding():

    audio = struct.pack(
        "<hhh",
        0,
        1000,
        -1000,
    )

    samples = struct.unpack(
        "<hhh",
        audio,
    )

    assert samples == (
        0,
        1000,
        -1000,
    )