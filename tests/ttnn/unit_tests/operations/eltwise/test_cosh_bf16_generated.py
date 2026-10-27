# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 cosh cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_cosh_with_sqrt_in_sfpu_chain(device, generated_first):
    """cosh in one SFPU chain with stock sqrt, both orders.

    Each chain step runs the kernel its own TT-NN op runs, and the second op stores exactly what it
    stores standalone on the first op's device result, which the chain keeps in DEST.
    """
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.COSH), lambda tensor: ttnn.cosh(tensor))
    stock = (ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT), lambda tensor: ttnn.sqrt(tensor))
    first, second = (generated, stock) if generated_first else (stock, generated)
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
    tt_in = to_tt_tensor(input_tensor, device)

    def words(tensor):
        return ttnn.to_torch(tensor).view(torch.int16)

    for step, standalone in (generated, stock):
        assert torch.equal(
            words(ttnn.unary_chain(tt_in, [step])), words(standalone(tt_in))
        ), f"{step} is not its op's kernel"

    result = words(ttnn.unary_chain(tt_in, [first[0], second[0]]))
    middle = ttnn.to_torch(first[1](tt_in))
    expected = words(second[1](to_tt_tensor(middle, device)))
    # A finite normal first result sits in DEST exactly as stored, so both runs feed the second op the same input.
    checked = torch.isfinite(middle) & (middle.abs() >= SMALLEST_NORMAL_BF16)
    assert checked.sum() >= 64
    mismatched = checked & (result != expected)
    assert not mismatched.any(), (
        f"{int(mismatched.sum())} of {int(checked.sum())} lanes differ from the standalone second op, "
        f"first at its input {middle[mismatched][0].item()!r}"
    )


def test_cosh_bf16_unflushed_zero_and_subnormal_inputs(device):
    """TT-NN's copy into DEST keeps -0 and subnormal BF16 inputs, which the LLK sweep's unpack delivers as +0.

    Each such input and +0 must store torch's result: its class as the BF16 pack stores it (NaN as
    +inf, -0 as +0, a subnormal result as 0) and a finite result within 1 ULP of torch's float64
    value. STOCK lists, per board, the inputs where the kernel stores its stock kernel's word
    instead, as {word: [(first input, last input), ...]} over the input bits.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.cosh keeps its stock kernel on {board}")
    STOCK = {}
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        t = torch.where(torch.isnan(t), torch.full_like(t, float("inf")), t)
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    actual = stored(ttnn.to_torch(ttnn.cosh(tt_x)))
    x64 = x.flatten().to(torch.float64)
    reference = ttnn.get_golden_function(ttnn.cosh)(x64).to(torch.float64).flatten()
    rounded = stored(reference)
    spacing = 2.0 ** (torch.floor(torch.log2(reference.abs().clamp(min=tiny))) - 7)
    close = torch.isfinite(reference) & (actual != 0) & ((actual - reference).abs() < spacing)
    ok = (actual == rounded) | close
    bits = words.repeat(4).to(torch.int32) & 0xFFFF
    for word, ranges in STOCK.get(board, {}).items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            ok |= (bits >= first) & (bits <= last) & (actual == value)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )


def test_cosh_bf16_bfloat8_b(device):
    """bfloat8_b in and out. The SFPU reads Bfp8_b data in DEST as BF16, so this runs the BF16 kernel, as the
    BF16 cases do. The result must correlate with torch's on the input as stored to PCC 0.999, as upstream's
    bfloat8_b unary cases require. Over every finite BF16 exponent, each output must also be within one bfloat8_b
    step of the BF16 call's for the same stored input, packed to bfloat8_b (two in a block that holds an infinity
    or NaN), and of its class where that is infinity or NaN: the op packs DEST to Bfp8_b, and the typecast
    rounds and sets a block's exponent its own way.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.cosh keeps its stock kernel on {board}")
    torch.manual_seed(0)
    x = torch.empty(1, 1, 64, 64).uniform_(-89.0, 89.0).to(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.float64).flatten()
    output = ttnn.cosh(tt_x)
    assert output.dtype == ttnn.bfloat8_b
    actual = ttnn.to_torch(output).to(torch.float64).flatten()
    golden = ttnn.get_golden_function(ttnn.cosh)(stored).to(torch.float64)
    pcc = torch.corrcoef(torch.stack([actual, golden]))[0, 1].item()
    assert pcc >= 0.999, f"PCC {pcc:.6f} against torch"
    # Every finite BF16 exponent and +-0, one exponent to each 16-value face row, so Bfp8_b keeps each value
    # to 7 bits under its own exponent.
    words = torch.arange(1 << 16, dtype=torch.int32)
    exponent = (words >> 7) & 0xFF
    zeros = torch.tensor([0x0000] * 256 + [0x8000] * 256, dtype=torch.int32)
    words = torch.cat([words[(exponent > 0) & (exponent < 0xFF)], zeros])
    x = words.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_x).to(torch.bfloat16)
    tt_stored = ttnn.from_torch(stored, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.cosh(tt_x)).to(torch.float32).flatten()
    expected = ttnn.to_torch(ttnn.typecast(ttnn.cosh(tt_stored), ttnn.bfloat8_b)).to(torch.float32).flatten()
    # One step of a Bfp8_b block: its 16 values share the largest one's exponent and keep 7 bits under it.
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    magnitude = torch.where(finite, torch.maximum(actual.abs(), expected.abs()), torch.zeros_like(actual))
    block = magnitude.reshape(-1, 16).amax(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.exp2(torch.floor(torch.log2(block.clamp_min(torch.finfo(torch.float32).tiny))) - 6)
    # A block that also holds an infinity or NaN: the op's pack and the typecast set its exponent apart, and
    # its finite values land up to two steps apart (polygamma's overflow binade, BH job 142644).
    special = (~finite).reshape(-1, 16).any(dim=1, keepdim=True).expand(-1, 16).reshape(-1)
    step = torch.where(special, 2 * step, step)
    # Infinity and NaN keep their class exactly; a finite value, zero included, may round one step apart (a value
    # under one step of its block packs to zero on one side and not the other: gelu at -8.625, BH job 142674).
    same = (
        (actual == expected)
        | (torch.isnan(actual) & torch.isnan(expected))
        | (finite & ((actual - expected).abs() <= step))
    )
    assert same.all(), (
        f"{int((~same).sum())} bfloat8_b outputs are more than one bfloat8_b step from the BF16 call's on the same "
        "stored input; first at "
        f"{stored.flatten()[~same][0].item()!r}: {actual[~same][0].item()!r}, BF16 call {expected[~same][0].item()!r}"
    )


def test_cosh_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs through ttnn.cosh, the call TT-NN users make.

    Each must store torch's float64 result (torch.cosh(x)) as the BF16 pack stores it (NaN as an
    infinity, a result below the smallest normal, or within 2^-12 above it, as 0), a finite word within
    STEPS BF16 steps of it, or the word STOCK lists for that input on the board, as
    {word: [(first input, last input), ...]} over the input bits. SKIP lists the inputs this case does
    not judge, which the LLK sweep checks against its own golden: where torch's float64 formula misses
    the exact result (a cancellation), where the kernel departs from torch by its spec, and NaN inputs
    whose stock word varies with the payload. MODEL holds every other normal input whose torch result is finite to the
    kernel's own word: torch's, moved by MODEL's whole BF16 steps where MODEL lists the input, as
    {steps: "first-last first ..."} (hex input words) for each board.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole", "wormhole_b0"):
        pytest.skip(f"ttnn.cosh keeps its stock kernel on {board}")
    STOCK = {}
    STEPS = 1
    SKIP = []
    MODEL = {"blackhole": {}, "wormhole_b0": {}}
    bits = torch.arange(1 << 16, dtype=torch.int32)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    def ordered(t):
        # BF16 words in value order, so that adjacent finite values differ by one step.
        words = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(words < 0, -(words & 0x7FFF), words)

    actual = stored(ttnn.to_torch(ttnn.cosh(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.cosh(x)).to(torch.float64).flatten()
    expected = stored(reference)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    ok = (actual == expected) | (finite & ((ordered(actual) - ordered(expected)).abs() <= STEPS))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(reference) & (torch.isnan(actual) | torch.isinf(actual))
    # The SFPU flushes a result below the smallest normal, or within 2^-12 above it, to 0.
    ok |= (actual == 0) & (reference.abs() < tiny * (1 + 2.0**-12))
    listed = torch.zeros_like(ok)
    for word, ranges in STOCK.get(board, {}).items():
        value = stored(torch.tensor([word], dtype=torch.int32).to(torch.int16).view(torch.bfloat16))
        for first, last in ranges:
            lanes = (bits >= first) & (bits <= last)
            listed |= lanes
            ok |= lanes & ((actual == value) | (torch.isnan(actual) & torch.isnan(value)))
    for first, last in SKIP:
        listed |= (bits >= first) & (bits <= last)
        ok |= (bits >= first) & (bits <= last)
    shift = torch.zeros_like(bits)
    for offset, spans in MODEL.get(board, {}).items():
        for span in spans.split():
            first, _, last = span.partition("-")
            shift[(bits >= int(first, 16)) & (bits <= int(last or first, 16))] = offset
    moved = ordered(expected) + shift
    word = torch.where(moved < 0, (-moved) | 0x8000, moved)
    held = stored(word.to(torch.int16).view(torch.bfloat16))
    normal = ((bits & 0x7F80) != 0) & ((bits & 0x7F80) != 0x7F80)
    ok = torch.where(normal & ~listed & torch.isfinite(expected), actual == held, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
