# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# Test fixture for tt-orchard's kernel loop: OPERATION loops forever in plain Python, so the check
# must end timeout when it is run with --timeout-s 5.
import ttl
import ttnn


def OPERATION(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    n = 0
    while True:
        n += 1
