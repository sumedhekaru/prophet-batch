# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fit and forecast many Prophet models at once, on top of stock Prophet."""

from __future__ import annotations

__version__ = '0.1.0.dev0'

from prophet_batch.predict import predict  # noqa: E402

__all__ = ['predict']
