# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest

from demo_hf_backend import ModelCache


def test_cache_hit_does_not_call_load_fn_again():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("a")
    assert calls == ["a"]


def test_cache_evicts_least_recently_used_when_full():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("b")
    cache.get("c")  # should evict "a", the least recently used
    assert cache.loaded_keys() == ("b", "c")


def test_touching_a_key_protects_it_from_eviction():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("b")
    cache.get("a")  # touch "a" again, making "b" the least recently used
    cache.get("c")  # should evict "b", not "a"
    assert cache.loaded_keys() == ("a", "c")


def test_get_on_an_evicted_key_reloads_it():
    calls = []
    cache = ModelCache(lambda k: calls.append(k) or k, maxsize=2)
    cache.get("a")
    cache.get("b")
    cache.get("c")  # evicts "a"
    cache.get("a")  # must reload
    assert calls == ["a", "b", "c", "a"]


def test_maxsize_must_be_at_least_one():
    with pytest.raises(ValueError):
        ModelCache(lambda k: k, maxsize=0)
